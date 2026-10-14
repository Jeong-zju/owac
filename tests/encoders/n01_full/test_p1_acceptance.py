"""Independent adversarial checks for the P1 hand-off, including failed controls."""

import math

import numpy as np
import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full import reference, replay
from owac.frameworks.n01_full.config import ReplayConfig
from owac.frameworks.n01_full.domains import canonical_sources, make_domain
from owac.frameworks.n01_full.ledger import ContentReplacement, MeasurementLedger
from owac.frameworks.n01_full.statistics import (
    CoordinateTransform,
    GaussianPrior,
    InterpolationSpec,
    NoiseModel,
    fit_structure,
)

from .test_replay import all_keys, builder_for, grid, observe_event, stream
from .test_statistics import affine_domain, constant_domain, line_keys, make_ledger


def test_unknown_correlation_affine_bound_cannot_cancel():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    fit = fit_structure(
        affine_domain(keys),
        ledger,
        noise_model=NoiseModel.unknown_correlation([1, 1], source_order=keys),
        dimension=1,
    )
    # At the first sample the interpolating affine estimate is exactly y_0.
    # Its variance is 1 for EVERY allowed correlation, not zero.
    assert fit.predict(0, (0, 0))[1] == pytest.approx(1.0)


def test_unknown_correlation_interpolation_bound_uses_raw_noise():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    spec = InterpolationSpec(((0.5, 0.0),), torch.tensor([[0.5, 0.5]], dtype=torch.float64))
    fit = fit_structure(
        constant_domain(keys),
        ledger,
        spec,
        NoiseModel.unknown_correlation([1, 1], source_order=keys),
    )
    # Perfectly correlated unit noise survives averaging with variance one.
    assert fit.predict(0, (0, 0))[1] == pytest.approx(1.0)


def test_unknown_correlation_rank_deficiency_has_no_false_covariance():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    fit = fit_structure(
        affine_domain(keys),
        ledger,
        noise_model=NoiseModel.unknown_correlation([1, 1], source_order=keys),
    )
    assert fit.estimate.covariance is None
    assert fit.estimate.uncertainty_kind == "none"


def test_unidentifiable_query_is_invalid_even_when_coefficients_are_finite():
    ledger = make_ledger([1.0, 3.0])
    fit = fit_structure(affine_domain(line_keys(2)), ledger)
    assert math.isnan(fit.predict(0, (0, 1))[0])
    assert fit.predict(0, (0, 0))[0] == pytest.approx(1.0)


def test_oracle_keeps_prior_in_unobserved_direction():
    # One measurement of theta_0; theta_1 comes only from the declared prior.
    result = reference.dense_gls(
        [[1, 0]], [1], [[1]], prior_mean=[0, 7], prior_covariance=np.eye(2)
    )
    np.testing.assert_allclose(result.theta, [0.5, 7])
    np.testing.assert_allclose(result.covariance, np.diag([0.5, 1]))


def test_noise_binding_cannot_be_silently_overridden():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
        noise_version="noise-a",
        prior_id="prior-a",
    )
    with pytest.raises(ValueError, match="noise"):
        fit_structure(
            domain,
            ledger,
            noise_model=NoiseModel.diagonal(dict.fromkeys(keys, 1), version="noise-b"),
        )
    with pytest.raises(ValueError, match="prior"):
        fit_structure(
            domain,
            ledger,
            noise_model=NoiseModel.diagonal(dict.fromkeys(keys, 1), version="noise-a"),
        )


def test_frame_read_api_cannot_mutate_raw_records():
    ledger = make_ledger([1.0, 3.0])
    ledger.frames()[0].values.fill_(999)
    ledger.get_frame("e0", "c0", 0).validity.fill_(1)
    assert ledger.get_source(line_keys(2)[0]).raw_value == 1
    assert ledger.validity_of(line_keys(2)[0]) == "valid"


def test_counterfactual_frame_api_and_overlay_cannot_resurrect_source():
    ledger = make_ledger([1.0, 3.0])
    key = line_keys(2)[0]
    removed = ledger.with_content(ContentReplacement("edit", {key: 100})).without([key])
    assert not removed.has_source(key)
    assert math.isnan(float(removed.values_for([key])[0]))
    frame = removed.frames()[0]
    assert frame.validity_of(0, 0, 0) == "missing"
    assert float(frame.values[0, 0, 0]) == 0.0


def test_unbatched_masks_follow_unbatched_values():
    ledger = MeasurementLedger("unbatched")
    ledger.observe(
        MeasurementFrame(
            values=torch.ones(1, 2, 2),
            valid_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
        )
    )
    assert ledger.ledger_stats()["validity_counts"]["valid"] == 4


def test_exact_window_actually_bounds_state_and_solve():
    result = replay.full_replay(
        stream(34), domain_builder=builder_for(all_keys(34)), readout_grid=grid()
    )
    assert result.ledger.frame_count() == 32
    assert len(result.history_loss) == 2
    assert result.fits[0].estimate.independent_observation_count == 64
    assert float(result.readout.means[0]) == pytest.approx(19.0)


def test_beyond_window_rebuild_drops_old_sources_not_just_a_label():
    keys = all_keys(10)
    result = replay.counterfactual_replay(
        stream(10),
        domain_builder=builder_for(keys),
        retracted_ids=[keys[0]],
        readout_grid=grid(),
        replay_config=ReplayConfig(exact_window_unique_frames=8),
    )
    assert result.result.ledger.frame_count() == 8
    assert result.result.fits[0].estimate.independent_observation_count == 16
    assert float(result.result.readout.means[0]) == pytest.approx(7.0)


def test_fixed_replay_rejects_empty_replacement_or_query():
    events = stream(2)
    ledger, _ = replay.consume(events)
    keys = all_keys(2)
    with pytest.raises(ValueError, match="replacement"):
        replay.fixed_decision_replay(
            events,
            frozen_domains=builder_for(keys)(ledger),
            retracted_ids=[keys[0]],
            replacement=ContentReplacement("empty", {}),
            readout_grid=grid(),
        )
    with pytest.raises(ValueError, match="query"):
        replay.fixed_decision_replay(
            events,
            frozen_domains=builder_for(keys)(ledger),
            retracted_ids=[keys[0]],
            replacement=ContentReplacement("edit", {keys[0]: 9}),
        )


def test_large_integer_identity_order_is_exact():
    from dataclasses import replace

    first = replace(line_keys(1)[0], frame_id=2**60)
    second = replace(first, frame_id=2**60 + 1)
    assert canonical_sources([second, first])[0] == (first, second)


def test_counterfactual_support_decision_cannot_read_deleted_value():
    keys = all_keys(2)

    def builder(ledger):
        frame = ledger.get_frame("e0", "c0", 0)
        # Deliberate content-sensitive support rule, using the public frame API.
        chosen = [keys[2]] if float(frame.values[0, 0, 0]) > 10 else [keys[3]]
        return builder_for(chosen)(ledger)

    outputs = []
    for deleted_value in (1.0, 100.0):
        events = replay.EventStream(
            "replay", (observe_event([deleted_value, 3], 0), observe_event([5, 9], 1))
        )
        outputs.append(
            float(
                replay.counterfactual_replay(
                    events, domain_builder=builder, retracted_ids=[keys[0]], readout_grid=grid()
                ).result.readout.means[0]
            )
        )
    assert outputs == [9, 9]


def test_affine_prior_stays_in_the_same_physical_chart_after_retraction():
    ledger = make_ledger([1.0, 3.0, 5.0])
    keys = line_keys(3)
    prior = GaussianPrior(
        "physical",
        torch.tensor([0.0, 2.0], dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        CoordinateTransform((0.0, 0.0), (1.0, 1.0), "ref"),
    )
    for selected in (keys, keys[1:]):
        fit = fit_structure(affine_domain(selected), ledger, dimension=1, prior=prior)
        design = np.array([[1.0, key.sensor_x] for key in selected])
        truth = reference.dense_gls(
            design,
            [1 + 2 * key.sensor_x for key in selected],
            prior_mean=[0.0, 2.0],
            prior_covariance=np.eye(2),
        )
        for x in (0.0, 1.5, 4.0):
            q = np.array([1.0, x])
            actual = fit.predict(0, (x, 0.0))
            np.testing.assert_allclose(
                actual, [q @ truth.theta, q @ truth.covariance @ q], atol=1e-10
            )


def test_unknown_correlation_bound_covers_signed_random_affine_queries():
    rng = np.random.default_rng(20261014)
    ledger = make_ledger(np.arange(6).reshape(2, 3))
    from owac.frameworks.n01_full.contracts import SourceKey

    keys = [SourceKey("stats", "e0", "c0", 0, 0, y, x) for y in range(2) for x in range(3)]
    variances = np.linspace(0.5, 2.0, 6)
    fit = fit_structure(
        affine_domain(keys),
        ledger,
        noise_model=NoiseModel.unknown_correlation(variances, source_order=keys),
    )
    design = np.array([[1.0, k.sensor_x, k.sensor_y] for k in keys])
    precision = np.diag(1 / variances)
    weights = np.linalg.solve(design.T @ precision @ design, design.T @ precision)
    for xy in rng.uniform(-3, 4, (30, 2)):
        a = np.array([1.0, *xy]) @ weights
        exact_bound = (np.abs(a) @ np.sqrt(variances)) ** 2
        assert fit.predict(0, tuple(xy))[1] == pytest.approx(exact_bound, abs=1e-10)


def test_counterfactual_recomputes_mapping_and_noise_on_redacted_content():
    keys = all_keys(2)
    calls = []

    def mapping(ledger):
        calls.append(("mapping", float(ledger.get_frame("e0", "c0", 0).values[0, 0, 0])))
        return None

    def noise(ledger):
        calls.append(("noise", float(ledger.get_frame("e0", "c0", 0).values[0, 0, 0])))
        return None

    replay.counterfactual_replay(
        stream(2),
        domain_builder=builder_for(keys),
        retracted_ids=[keys[0]],
        readout_grid=grid(),
        correspondence_builder=mapping,
        noise_model_builder=noise,
    )
    assert calls == [("mapping", 0.0), ("noise", 0.0)]


def test_policy_gradient_checks_do_not_hide_detached_encoder_behind_bridge():
    from owac.runtime.n01_full_policy_preflight import gradient_receipt

    detached = torch.nn.Parameter(torch.ones(2))
    bridge = torch.nn.Parameter(torch.ones(2))
    bridge.sum().backward()
    assert gradient_receipt([detached]) == {"total": 0, "finite": False, "tensors": 0}
    assert gradient_receipt([bridge])["total"] > 0


def test_separate_phase_fits_preserve_cross_phase_error_covariance():
    ledger = make_ledger(torch.tensor([[[[1.0]], [[3.0]]]], dtype=torch.float64))
    keys = [line_keys(1, channel=i)[0] for i in (0, 1)]
    sigma = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    fit = fit_structure(
        constant_domain(keys, phase=None), ledger, noise_model=NoiseModel.dense(keys, sigma)
    )
    torch.testing.assert_close(fit.estimate.covariance, sigma)
    proxy = fit_structure(
        constant_domain(keys, phase=None),
        ledger,
        noise_model=NoiseModel.unknown_correlation([1.0, 1.0], source_order=keys),
    )
    # The sum of the two phases has variance four for perfectly shared noise.
    q = torch.ones(2, dtype=torch.float64)
    assert float(q @ proxy.estimate.covariance @ q) >= 4.0


def test_missing_content_overlay_is_consistent_across_public_queries():
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    view = ledger.with_content(ContentReplacement("missing", {keys[0]: None}))
    assert not view.has_source(keys[0])
    assert view.frames()[0].validity_of(0, 0, 0) == "missing"
    assert math.isnan(float(view.values_for([keys[0]])[0]))
    assert fit_structure(constant_domain(keys), view).predict(0, (0, 0))[0] == pytest.approx(3.0)


@pytest.mark.parametrize("variance", [1e-30, 1e30])
def test_noise_unit_rescaling_does_not_change_identifiability(variance):
    ledger = make_ledger([1.0, 3.0])
    keys = line_keys(2)
    fit = fit_structure(
        constant_domain(keys), ledger, noise_model=NoiseModel.dense(keys, np.eye(2) * variance)
    )
    assert fit.rank == 1
    assert fit.predict(0, (0, 0))[0] == pytest.approx(2.0)
    assert fit.predict(0, (0, 0))[1] / variance == pytest.approx(0.5)

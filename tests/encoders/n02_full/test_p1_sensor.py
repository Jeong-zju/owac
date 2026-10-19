"""Measurement-density tests for the N02 P1 sensor module (T2).

Every test here is meant to fail on a real defect: a missing ``log(v)`` penalty, an
unstable quantisation/density term, a masked sample leaking into the score, a unit
mismatch being scored anyway, an oracle/truth field reaching the history eta estimator,
or a duplicated sample identity being counted twice.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from statistics import NormalDist

import numpy as np
import pytest
import torch

from owac.frameworks.n02_full import sensor as sensor_mod
from owac.frameworks.n02_full import synthetic as synth
from owac.frameworks.n02_full.sensor import (
    DensityVerificationError,
    InterferenceEstimate,
    InterferenceObservation,
    InterferencePrior,
    MeasurementScore,
    RawPrediction,
    SensorModel,
    UnitsMismatchError,
)

DTYPE = torch.float64


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _top_tail(mean: float, variance: float, level: float) -> float:
    return 0.5 * math.erfc((level - mean) / math.sqrt(2.0 * variance))


def _score(prediction: RawPrediction, values, **kwargs) -> MeasurementScore:
    return sensor_mod.score_prediction(prediction, values, units="dn", **kwargs)


def test_frozen_dataclass_fields_are_the_declared_interface():
    assert [item.name for item in dataclasses.fields(RawPrediction)][:4] == [
        "mean",
        "variance",
        "masks",
        "units",
    ]
    assert [item.name for item in dataclasses.fields(MeasurementScore)] == [
        "relative_terms",
        "raw_nll_or_energy",
        "local_stats",
        "kind",
        "per_valid_scalar",
    ]
    assert [item.name for item in dataclasses.fields(InterferenceEstimate)] == [
        "log_exposure",
        "camera_translation",
        "noise_scale",
        "source",
    ]


def test_gaussian_nll_matches_fp64_closed_form_elementwise():
    values = torch.tensor([0.3, -1.2, 4.0, 0.0], dtype=DTYPE)
    mean = torch.tensor([0.1, 0.0, 1.0, 0.0], dtype=DTYPE)
    variance = torch.tensor([0.5, 2.0, 0.01, 7.5], dtype=DTYPE)
    got = sensor_mod.gaussian_nll(values, mean, variance)
    closed_form = 0.5 * (
        (values - mean) ** 2 / variance + torch.log(variance) + math.log(2.0 * math.pi)
    )
    assert torch.allclose(got, closed_form, rtol=0.0, atol=1e-14)
    reference = torch.tensor(
        [
            -math.log(NormalDist(float(m), math.sqrt(float(v))).pdf(float(y)))
            for y, m, v in zip(values, mean, variance, strict=True)
        ],
        dtype=DTYPE,
    )
    assert torch.allclose(got, reference, rtol=0.0, atol=1e-12)
    # float32 agrees with the float64 reference to float32 resolution
    got32 = sensor_mod.gaussian_nll(values.float(), mean.float(), variance.float()).to(DTYPE)
    relative = (got32 - reference).abs() / reference.abs().clamp_min(1.0)
    assert float(relative.max()) < 1e-5


def test_quantisation_log_prob_matches_fp64_numerical_integration():
    cases = [
        (0.0, 0.0, 1.0),
        (1.0, 0.5, 0.3),
        (-3.0, 0.0, 2.0),
        (7.5, 7.0, 1.0),
        (2.0, 0.0, 10.0),
        (-0.25, 0.25, 0.05),
    ]
    for value, mean, sigma in cases:
        got = sensor_mod.quantisation_log_prob(
            torch.tensor([value], dtype=DTYPE),
            torch.tensor([mean], dtype=DTYPE),
            torch.tensor([sigma * sigma], dtype=DTYPE),
            step=1.0,
        )
        lower = (value - 0.5 - mean) / sigma
        upper = (value + 0.5 - mean) / sigma
        expected = math.log(_normal_cdf(upper) - _normal_cdf(lower))
        assert float((got[0] - expected).abs()) < 1e-10, (value, mean, sigma)
        # the numpy declaration in synthetic.py is the same interval probability
        declared = sensor_mod.declared_quantisation_interval_probability(
            value, mean, sigma * sigma, step=1.0
        )
        # the numpy declaration uses an A&S normal approximation (about 1e-7 absolute)
        assert math.isclose(math.exp(float(got[0])), float(declared), rel_tol=1e-5)


def test_quantisation_log_prob_stays_finite_in_deep_tails_and_wide_intervals():
    # deep tail: the interval probability must not cancel to log(0)
    deep = sensor_mod.quantisation_log_prob(
        torch.tensor([-30.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([1.0], dtype=DTYPE),
        step=1.0,
    )
    assert torch.isfinite(deep).all()
    assert float(deep[0]) < -100.0
    # tiny variance: an interval far wider than sigma has probability ~1, NLL ~0
    tiny = sensor_mod.quantisation_log_prob(
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([1e-24], dtype=DTYPE),
        step=1.0,
    )
    assert float(tiny[0].abs()) < 1e-9
    # wide variance: NLL approaches log(sigma * sqrt(2 pi) / step)
    wide = sensor_mod.quantisation_log_prob(
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([0.0], dtype=DTYPE),
        torch.tensor([1e12], dtype=DTYPE),
        step=1.0,
    )
    expected = math.log(1e6 * math.sqrt(2.0 * math.pi))
    assert math.isclose(float(-wide[0]), expected, rel_tol=1e-9)


def test_clip_tail_log_probability_matches_declared_probability():
    mean = torch.tensor([0.1, -0.4, 1.0], dtype=DTYPE)
    variance = torch.tensor([0.5, 2.0, 0.01], dtype=DTYPE)
    level = 1.0
    top = sensor_mod.clip_tail_log_prob(mean, variance, level=level, side="top")
    bottom = sensor_mod.clip_tail_log_prob(mean, variance, level=level, side="bottom")
    for index in range(3):
        mu = float(mean[index])
        var = float(variance[index])
        assert math.isclose(
            float(top[index]), math.log(_top_tail(mu, var, level)), rel_tol=1e-12, abs_tol=1e-300
        )
        assert math.isclose(
            float(bottom[index]),
            math.log(1.0 - _top_tail(mu, var, level)),
            rel_tol=1e-12,
            abs_tol=1e-300,
        )
        declared = sensor_mod.declared_saturation_tail_probability(mu, var, level=level)
        assert math.isclose(math.exp(float(top[index])), float(declared), rel_tol=1e-6)


def test_score_prediction_uses_interval_probability_and_clip_tails():
    level = 1.0
    values = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=DTYPE)
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE),
        variance=torch.full((4,), 0.25, dtype=DTYPE),
        masks={"saturated": torch.tensor([False, False, True, False])},
        units="dn",
    )
    score = _score(prediction, values, quantisation_step=0.5, saturation_level=level)
    assert score.kind == "proper_nll"
    assert set(score.relative_terms) == {"quantisation", "clip_top"}
    # the clipped sample contributes the top tail, not a Gaussian/interval term
    clipped_log_probability = sensor_mod.clip_tail_log_prob(
        torch.zeros(1, dtype=DTYPE), torch.full((1,), 0.25, dtype=DTYPE), level=level, side="top"
    )
    clipped_nll = -clipped_log_probability
    assert math.isclose(
        float(score.relative_terms["clip_top"][2]), float(clipped_nll[0]), rel_tol=1e-12
    )
    assert math.isclose(
        float(score.local_stats["top_clip_tail_probability_max"]),
        float(torch.exp(clipped_log_probability[0])),
        rel_tol=1e-12,
    )
    assert float(score.local_stats["n_clipped_top"]) == 1.0
    assert float(score.local_stats["n_valid_scalars"]) == 4.0
    interval_nll = -sensor_mod.quantisation_log_prob(
        values, torch.zeros(4, dtype=DTYPE), torch.full((4,), 0.25, dtype=DTYPE), step=0.5
    )
    expected = interval_nll[0] + interval_nll[1] + interval_nll[3] + clipped_nll[0]
    assert math.isclose(float(score.raw_nll_or_energy), float(expected), rel_tol=1e-12)
    assert math.isclose(float(score.per_valid_scalar), float(expected) / 4.0, rel_tol=1e-12)


def test_score_prediction_bottom_clip_uses_the_lower_tail_and_refuses_undeclared_clips():
    values = torch.tensor([0.0, -1.0, -2.0], dtype=DTYPE)
    prediction = RawPrediction(
        mean=torch.zeros(3, dtype=DTYPE),
        variance=torch.full((3,), 0.25, dtype=DTYPE),
        masks={"saturated": torch.tensor([True, False, False])},
        units="dn",
    )
    # a saturated sample with no declared clip level cannot be scored as an exact value
    with pytest.raises(ValueError):
        _score(prediction, values, quantisation_step=0.5)
    score = _score(prediction, values, quantisation_step=0.5, floor_level=0.0)
    assert set(score.relative_terms) == {"quantisation", "clip_bottom"}
    bottom_log_probability = sensor_mod.clip_tail_log_prob(
        torch.zeros(1, dtype=DTYPE), torch.full((1,), 0.25, dtype=DTYPE), level=0.0, side="bottom"
    )
    assert math.isclose(
        float(score.relative_terms["clip_bottom"][0]),
        float(-bottom_log_probability[0]),
        rel_tol=1e-12,
    )
    assert math.isclose(
        float(score.local_stats["bottom_clip_tail_probability_max"]),
        float(torch.exp(bottom_log_probability[0])),
        rel_tol=1e-12,
    )
    assert float(score.local_stats["n_clipped_bottom"]) == 1.0
    assert float(score.local_stats["n_clipped_top"]) == 0.0


def test_zero_tiny_negative_and_infinite_variance_are_finite():
    values = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=DTYPE)
    variance = torch.tensor([0.0, -1e-6, 1e-300, float("inf")], dtype=DTYPE)
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE), variance=variance, masks={}, units="dn"
    )
    score = _score(prediction, values)
    assert torch.isfinite(score.raw_nll_or_energy)
    assert torch.isfinite(score.per_valid_scalar)
    assert bool(torch.isfinite(score.relative_terms["continuous"]).all())
    assert float(score.local_stats["n_variance_floored"]) == 4.0
    assert float(score.local_stats["min_variance_used"]) >= sensor_mod.MIN_VARIANCE
    assert float(score.local_stats["max_variance_used"]) <= sensor_mod.MAX_VARIANCE


def test_extreme_saturation_values_stay_finite_in_float64_and_float32():
    for dtype in (DTYPE, torch.float32):
        values = torch.tensor([65535.0, 65535.0], dtype=dtype)
        prediction = RawPrediction(
            mean=torch.zeros(2, dtype=dtype),
            variance=torch.full((2,), 1e-20, dtype=dtype),
            masks={},
            units="dn",
        )
        score = _score(prediction, values, quantisation_step=1.0)
        assert torch.isfinite(score.raw_nll_or_energy)
        assert bool(torch.isfinite(score.relative_terms["quantisation"]).all())
        # an extreme top clip through the tail probability is finite as well
        clipped = _score(prediction, values, saturation_level=65535.0)
        assert torch.isfinite(clipped.raw_nll_or_energy)


def test_huge_variance_cannot_improve_a_proper_score_for_free():
    value = torch.tensor([0.0], dtype=DTYPE)
    wrong_mean = torch.tensor([10.0], dtype=DTYPE)
    # fixed squared error 100: the score must grow monotonically past the optimum v=100
    variances = torch.tensor([100.0, 1e3, 1e6, 1e9, 1e12], dtype=DTYPE)
    nll = sensor_mod.gaussian_nll(value, wrong_mean, variances)
    assert bool((nll[1:] > nll[:-1]).all()), nll.tolist()
    assert float(nll[-1]) > 0.5 * math.log(1e12)
    # a wrong candidate cannot buy its way past a correct tight one by inflating variance
    correct = sensor_mod.gaussian_nll(
        value, torch.zeros(1, dtype=DTYPE), torch.ones(1, dtype=DTYPE)
    )
    cheat = sensor_mod.gaussian_nll(value, wrong_mean, torch.full((1,), 1e12, dtype=DTYPE))
    assert float(correct[0]) < float(cheat[0])
    # the quantised model penalises an inflated sigma in the same way
    sigmas = torch.tensor([1e2, 1e4, 1e6, 1e12], dtype=DTYPE)
    quantised = -sensor_mod.quantisation_log_prob(
        value, torch.zeros(4, dtype=DTYPE), sigmas * sigmas, step=1.0
    )
    assert bool((quantised[1:] > quantised[:-1]).all()), quantised.tolist()


def test_missing_and_bad_samples_contribute_no_measurement_term():
    values = torch.tensor([0.3, -0.7, 2.0, 1.1], dtype=DTYPE)
    masks = {
        "missing": torch.tensor([True, False, False, False]),
        "bad": torch.tensor([False, False, False, True]),
    }
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE), variance=torch.ones(4, dtype=DTYPE), masks={}, units="dn"
    )
    score = _score(prediction, values, masks=masks)
    assert float(score.local_stats["n_valid_scalars"]) == 2.0
    # garbage at a masked position cannot move the score
    garbage = values.clone()
    garbage[0] = 123.0
    garbage[3] = -999.0
    score_garbage = _score(prediction, garbage, masks=masks)
    assert torch.allclose(score.raw_nll_or_energy, score_garbage.raw_nll_or_energy, rtol=0, atol=0)
    # all missing/bad contributes nothing at all
    dead = _score(
        prediction,
        values,
        masks={"missing": torch.ones(4, dtype=torch.bool)},
    )
    assert float(dead.raw_nll_or_energy) == 0.0
    assert float(dead.per_valid_scalar) == 0.0
    assert float(dead.local_stats["n_valid_scalars"]) == 0.0


def test_nonfinite_values_are_not_measurements():
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE), variance=torch.ones(4, dtype=DTYPE), masks={}, units="dn"
    )
    values = torch.tensor([0.0, float("inf"), float("nan"), 1.0], dtype=DTYPE)
    score = _score(prediction, values)
    assert float(score.local_stats["n_nonfinite_values"]) == 2.0
    assert float(score.local_stats["n_valid_scalars"]) == 2.0
    two_sample = RawPrediction(
        mean=torch.zeros(2, dtype=DTYPE), variance=torch.ones(2, dtype=DTYPE), masks={}, units="dn"
    )
    finite_only = _score(two_sample, torch.tensor([0.0, 1.0], dtype=DTYPE))
    assert math.isclose(
        float(score.raw_nll_or_energy), float(finite_only.raw_nll_or_energy), rel_tol=1e-12
    )


def test_units_must_match_verbatim():
    prediction = RawPrediction(
        mean=torch.zeros(2, dtype=DTYPE), variance=torch.ones(2, dtype=DTYPE), masks={}, units="dn"
    )
    with pytest.raises(UnitsMismatchError):
        sensor_mod.score_prediction(
            prediction, torch.zeros(2, dtype=DTYPE), units="normalized_float"
        )
    with pytest.raises(UnitsMismatchError):
        sensor_mod.score_prediction(prediction, torch.zeros(2, dtype=DTYPE), units="DN")
    # a query object declares its own unit and needs no repeated argument
    query = type("Query", (), {})()
    query.values = np.zeros(2, dtype=np.float64)
    query.unit = "dn"
    score = sensor_mod.score_prediction(prediction, query)
    assert float(score.local_stats["n_valid_scalars"]) == 2.0
    # an undeclared query unit is refused instead of assumed
    with pytest.raises(ValueError):
        sensor_mod.score_prediction(prediction, torch.zeros(2, dtype=DTYPE))


def test_duplicate_sample_identity_is_counted_once():
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE), variance=torch.ones(4, dtype=DTYPE), masks={}, units="dn"
    )
    values = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=DTYPE)
    score = _score(prediction, values, sample_ids=["a", "a", "b", "b"])
    assert float(score.local_stats["n_duplicate_samples"]) == 2.0
    assert float(score.local_stats["n_valid_scalars"]) == 2.0
    single = 0.5 * (0.25 + math.log(2.0 * math.pi))
    assert math.isclose(float(score.raw_nll_or_energy), 2.0 * single, rel_tol=1e-12)
    # the same sensor position seen at two overlapping scales is one position
    coordinates = torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]], dtype=DTYPE)
    by_position = _score(prediction, values, coordinates=coordinates)
    assert float(by_position.local_stats["n_duplicate_samples"]) == 2.0
    assert float(by_position.local_stats["n_valid_scalars"]) == 2.0
    assert float(by_position.raw_nll_or_energy) == float(score.raw_nll_or_energy)
    # a duplicate identity with a mismatched length is refused, not silently padded
    with pytest.raises(ValueError):
        _score(prediction, values, sample_ids=["a", "b"])


def test_positive_variance_parameterisation_and_sensor_variance_components():
    raw = torch.tensor([-5.0, 0.0, 5.0], dtype=DTYPE)
    with pytest.raises(ValueError):
        sensor_mod.positive_variance(raw, parameterisation="unknown")
    for name in ("softplus", "exp"):
        variance = sensor_mod.positive_variance(raw, parameterisation=name)
        assert bool((variance >= sensor_mod.MIN_VARIANCE).all())
        assert torch.isfinite(variance).all()
    mean = torch.tensor([0.0, 0.5, 1.0], dtype=DTYPE)
    sensor_var = sensor_mod.sensor_variance(
        mean, shot_noise_scale=1000.0, read_noise_signal=0.01, noise_scale=2.0
    )
    expected = 4.0 * (mean / 1000.0 + 1e-4)
    assert torch.allclose(sensor_var, expected, rtol=1e-12, atol=1e-15)
    model = SensorModel(
        units="dn", shot_noise_scale=1000.0, read_noise_signal=0.01, noise_scale=2.0
    )
    prediction = model.build_prediction(
        mean, variance_parameter=torch.tensor([-1.0, 0.0, 1.0], dtype=DTYPE)
    )
    components = prediction.variance_components
    assert set(components) == {"sensor", "model", "total"}
    assert torch.allclose(
        components["total"], components["sensor"] + components["model"], rtol=1e-12, atol=1e-15
    )
    assert torch.allclose(components["sensor"], expected, rtol=1e-12, atol=1e-15)
    assert bool((components["model"] > 0.0).all())
    assert torch.allclose(prediction.variance, components["total"], rtol=0.0, atol=0.0)
    with pytest.raises(UnitsMismatchError):
        model.build_prediction(mean, units="other")


def test_empirical_energy_is_declared_for_unverified_pseudo_raw():
    model = SensorModel(units="pseudo_raw_dn_float32", density_verified=False)
    assert model.kind == "empirical_energy"
    prediction = model.build_prediction(
        torch.zeros(3, dtype=DTYPE), variance_parameter=torch.zeros(3, dtype=DTYPE)
    )
    values = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)
    score = model.score(prediction, values, units="pseudo_raw_dn_float32")
    assert score.kind == "empirical_energy"
    assert "energy" in score.relative_terms
    assert torch.isfinite(score.raw_nll_or_energy)
    # an unverified density can never be reported as a proper NLL
    with pytest.raises(DensityVerificationError):
        model.score(prediction, values, units="pseudo_raw_dn_float32", kind="proper_nll")


def test_local_stats_report_whole_map_mean_and_local_maximum():
    prediction = RawPrediction(
        mean=torch.zeros(4, dtype=DTYPE), variance=torch.ones(4, dtype=DTYPE), masks={}, units="dn"
    )
    values = torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=DTYPE)
    score = _score(prediction, values)
    assert math.isclose(
        float(score.local_stats["abs_standardised_residual_mean"]), 0.75, rel_tol=1e-12
    )
    assert math.isclose(
        float(score.local_stats["abs_standardised_residual_max"]), 3.0, rel_tol=1e-12
    )
    residual = score.local_stats["standardised_residual"]
    assert bool((residual == torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=DTYPE)).all())


def test_raw_prediction_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        RawPrediction(
            mean=torch.zeros(3, dtype=DTYPE),
            variance=torch.ones(2, dtype=DTYPE),
            masks={},
            units="dn",
        )
    with pytest.raises(ValueError):
        MeasurementScore(
            relative_terms={},
            raw_nll_or_energy=torch.tensor(0.0, dtype=DTYPE),
            local_stats={},
            kind="not_a_real_score_kind",
            per_valid_scalar=torch.tensor(0.0, dtype=DTYPE),
        )


def test_history_eta_never_reads_truth_fields():
    @dataclasses.dataclass
    class EvaluationTruth:
        frame_index: int = 0
        camera_id: str = "cam0"
        truth_log_exposure: float = 0.0
        truth_camera_translation: tuple[float, float] = (0.0, 0.0)

    with pytest.raises(ValueError):
        sensor_mod.estimate_interference([EvaluationTruth()])
    with pytest.raises(ValueError):
        sensor_mod.estimate_interference(
            [{"frame_index": 0, "camera_id": "cam0", "truth_log_exposure": 0.0}]
        )
    # the signature itself carries no truth/oracle/future channel
    parameter_names = list(inspect.signature(sensor_mod.estimate_interference).parameters)
    assert not any(
        token in name.lower()
        for name in parameter_names
        for token in ("truth", "oracle", "label", "future", "target", "evaluation")
    )
    forbidden = [
        name
        for name in sensor_mod.estimate_interference.__code__.co_names
        if any(token in name.lower() for token in ("truth", "oracle", "evaluation"))
    ]
    assert forbidden == []


def test_history_eta_estimates_source_and_clamps_to_the_frozen_prior():
    prior = InterferencePrior(
        log_exposure_range=(-1.0, 1.0),
        translation_range_px=(-2.0, 2.0),
        noise_scale_range=(0.5, 2.0),
    )
    history = [
        InterferenceObservation(
            0, "cam0", log_exposure=5.0, camera_translation=(10.0, -10.0), noise_scale=8.0
        ),
        InterferenceObservation(
            0, "cam0", log_exposure=5.0, camera_translation=(6.0, -6.0), noise_scale=8.0
        ),
        InterferenceObservation(1, "cam1", log_exposure=5.0, camera_translation=(0.5, -0.5)),
    ]
    estimate = sensor_mod.estimate_interference(
        history, prior=prior, frame_indices=(0, 1), camera_ids=("cam0", "cam1")
    )
    assert estimate.source == "history"
    assert float(estimate.log_exposure) == 1.0
    assert estimate.camera_translation.shape == (2, 2, 2)
    assert estimate.camera_translation[0, 0].tolist() == [2.0, -2.0]
    assert estimate.camera_translation[1, 1].tolist() == [0.5, -0.5]
    assert estimate.camera_translation[1, 0].tolist() == [0.0, 0.0]
    assert float(estimate.noise_scale) == 2.0
    # an empty history falls back to the frozen prior itself
    empty = sensor_mod.estimate_interference([], prior=prior)
    assert empty.source == "history"
    assert float(empty.log_exposure) == prior.log_exposure_mean
    assert float(empty.noise_scale) == prior.noise_scale_prior
    with pytest.raises(ValueError):
        InterferenceObservation(0, "cam0", log_exposure=math.inf)


def test_oracle_interference_is_a_separate_evaluator_side_construction():
    oracle = InterferenceEstimate(
        log_exposure=torch.tensor(0.5, dtype=DTYPE),
        camera_translation=torch.zeros(1, 1, 2, dtype=DTYPE),
        noise_scale=torch.tensor(1.25, dtype=DTYPE),
        source="oracle",
    )
    assert oracle.source == "oracle"
    estimate = sensor_mod.estimate_interference(
        [InterferenceObservation(0, "cam0", log_exposure=0.25)]
    )
    assert estimate.source == "history"
    with pytest.raises(ValueError):
        InterferenceEstimate(
            log_exposure=torch.tensor(0.0, dtype=DTYPE),
            camera_translation=torch.zeros(1, 1, 2, dtype=DTYPE),
            noise_scale=torch.tensor(1.0, dtype=DTYPE),
            source="guessed",
        )


def test_camera_motion_moves_every_projection_entity_motion_only_its_own():
    coordinates = torch.zeros(4, 2, dtype=DTYPE)
    translation = torch.tensor([1.0, -2.0], dtype=DTYPE)
    camera = sensor_mod.project_camera_motion(coordinates, translation)
    assert camera.tolist() == [[1.0, -2.0]] * 4
    entity = sensor_mod.project_entity_motion(
        coordinates,
        translation,
        entity_mask=torch.tensor([True, False, True, False]),
    )
    assert entity.tolist() == [[1.0, -2.0], [0.0, 0.0], [1.0, -2.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        sensor_mod.project_entity_motion(
            coordinates, translation, entity_mask=torch.tensor([True, False])
        )


def test_sensor_model_requires_a_declared_query_unit_for_bare_tensors():
    model = SensorModel(units="dn", shot_noise_scale=100.0)
    prediction = model.build_prediction(torch.zeros(2, dtype=DTYPE))
    with pytest.raises(ValueError):
        model.score(prediction, torch.zeros(2, dtype=DTYPE))
    score = model.score(prediction, torch.zeros(2, dtype=DTYPE), units="dn")
    assert torch.isfinite(score.raw_nll_or_energy)


# --------------------------------------------------------------------------------------
# unit/scale consistency on a 12-bit DN sensor (frozen P1 section 3.8)
# --------------------------------------------------------------------------------------

CFA_BLACK = 64.0
CFA_WHITE = 4095.0
CFA_SCALE = CFA_WHITE - CFA_BLACK
CFA_SHOT_NORMALISED = 1000.0
CFA_READ_E = 2.0
CFA_BACKGROUND = 0.4
CFA_READ_NORMALISED = CFA_READ_E / CFA_SHOT_NORMALISED
# the same declaration converted once into the declared DN units (factors-side affine):
#   shot_noise_scale_units = shot_noise_scale_normalised / scale
#   read_noise_signal_units = read_noise_signal_normalised * scale
CFA_SHOT_UNITS = CFA_SHOT_NORMALISED / CFA_SCALE
CFA_READ_UNITS = CFA_READ_NORMALISED * CFA_SCALE
CFA_UNIT = "raw_dn_uint16_mosaic"


def _dn_sensor_model(**overrides) -> SensorModel:
    declared = {
        "units": CFA_UNIT,
        "quantisation_step": 1.0,
        "saturation_level": CFA_WHITE,
        "floor_level": CFA_BLACK,
    }
    declared.update(overrides)
    return SensorModel(**declared)


def _cfa_flat_measurement(seed: int = 17):
    """One real synthetic 12-bit CFA flat-field measurement (DN, black 64, white 4095)."""
    sensor = synth.SensorSpec(
        camera_id="cam0",
        height=8,
        width=8,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
        black_level=CFA_BLACK,
        white_level=CFA_WHITE,
        bit_depth=12,
        read_noise_e=CFA_READ_E,
        gain=1.0,
        exposure_ms=1.0,
    )
    model = synth.MeasurementModel(
        kind="cfa",
        shot_noise_scale=CFA_SHOT_NORMALISED,
        read_noise_e=CFA_READ_E,
        saturation_level=1.0,
        response=(0.95, 1.0, 1.0, 0.75),
        bad_pixel_fraction=0.0,
        missing_sample_fraction=0.0,
    )
    state = synth.RenderState(
        frame_index=0,
        shapes=(),
        sensors=(sensor,),
        background_level=CFA_BACKGROUND,
        illumination=1.0,
        exposure_scale=1.0,
        shot_noise_scale=CFA_SHOT_NORMALISED,
        read_noise_e=CFA_READ_E,
        read_noise_e_source="sensor_read_noise_e",
        camera_shift_px={},
        noise_seed=seed,
    )
    return model.render(state)["cam0"]


def test_sensor_model_declares_every_parameter_in_the_query_units():
    model = _dn_sensor_model(shot_noise_scale=CFA_SHOT_UNITS, read_noise_signal=CFA_READ_UNITS)
    semantics = model.unit_semantics()
    assert semantics["units"] == CFA_UNIT
    assert semantics["variance_unit"] == f"{CFA_UNIT}^2"
    assert semantics["quantisation_step"] == 1.0
    assert semantics["quantisation_step_unit"] == CFA_UNIT
    assert semantics["saturation_level"] == CFA_WHITE
    assert semantics["saturation_level_unit"] == CFA_UNIT
    assert semantics["floor_level"] == CFA_BLACK
    assert semantics["floor_level_unit"] == CFA_UNIT
    assert semantics["min_variance_unit"] == f"{CFA_UNIT}^2"
    assert semantics["max_variance_unit"] == f"{CFA_UNIT}^2"
    assert semantics["noise_scale_unit"] == "dimensionless_sigma_multiplier"
    assert semantics["shot_noise_scale_unit"] == f"electrons_per_{CFA_UNIT}_of_declared_signal"
    assert semantics["read_noise_signal_unit"] == CFA_UNIT
    assert semantics["normalisation"] == "none_in_sensor_unit_affine_is_declared_by_factors"
    # no normalised-domain field survives: the affine lives on the factors side
    field_names = {item.name for item in dataclasses.fields(SensorModel)}
    assert not {"black_level", "white_level", "signal_scale"} & field_names
    assert "oracle_prediction" not in dir(sensor_mod)
    # the declared shot/read statement is the plain declared-unit formula
    signal_dn = torch.tensor([100.0, 500.0], dtype=DTYPE)
    expected = signal_dn / CFA_SHOT_UNITS + CFA_READ_UNITS**2
    assert torch.allclose(model.sensor_variance(signal_dn), expected, rtol=1e-12)


def test_dn_prediction_scores_o1_with_finite_gradients():
    """mean = 2048 DN, variance = 4 DN**2, observations within +/-2 DN: O(1) NLL."""
    model = _dn_sensor_model()
    values = torch.tensor([2046.0, 2050.0, 2048.0], dtype=DTYPE)
    for dtype in (torch.float32, torch.float64):
        mean = torch.full((3,), 2048.0, dtype=dtype, requires_grad=True)
        prediction = RawPrediction(
            mean=mean,
            variance=torch.full((3,), 4.0, dtype=dtype),
            masks={},
            units=CFA_UNIT,
        )
        score = model.score(prediction, values.to(dtype), units=CFA_UNIT)
        diagnostics = model.diagnostics(prediction, values.to(dtype), units=CFA_UNIT)
        score.raw_nll_or_energy.backward()
        per_scalar = float(score.per_valid_scalar.detach())
        assert 0.5 < per_scalar < 5.0, per_scalar
        assert float(score.local_stats["n_valid_scalars"].detach()) == 3.0
        assert bool(torch.isfinite(mean.grad).all())
        assert diagnostics["scale_suspect"] is False
        assert diagnostics["variance_median"] == pytest.approx(4.0)


def test_normalised_domain_prediction_is_rejected_or_scores_astronomically():
    """The identical model with normalised-domain numbers must not score well."""
    model = _dn_sensor_model()
    values = torch.tensor([2046.0, 2050.0, 2048.0], dtype=DTYPE)
    correct = RawPrediction(
        mean=torch.full((3,), 2048.0, dtype=DTYPE),
        variance=torch.full((3,), 4.0, dtype=DTYPE),
        masks={},
        units=CFA_UNIT,
    )
    correct_nll = float(model.score(correct, values, units=CFA_UNIT).per_valid_scalar)
    assert correct_nll < 5.0

    # (a) a different declared unit is refused before any comparison
    mismatched = RawPrediction(
        mean=torch.full((3,), 2048.0 / CFA_SCALE, dtype=DTYPE),
        variance=torch.full((3,), 4.0 / CFA_SCALE**2, dtype=DTYPE),
        masks={},
        units="normalised_signal_float32",
    )
    with pytest.raises(UnitsMismatchError):
        model.score(mismatched, values, units=CFA_UNIT)

    # (b) normalised-domain numbers under the DN label: a silent scale bug explodes
    silent = RawPrediction(
        mean=torch.full((3,), 2048.0 / CFA_SCALE, dtype=DTYPE),
        variance=torch.full((3,), 4.0 / CFA_SCALE**2, dtype=DTYPE),
        masks={},
        units=CFA_UNIT,
    )
    silent_score = model.score(silent, values, units=CFA_UNIT)
    silent_diagnostics = model.diagnostics(silent, values, units=CFA_UNIT)
    assert float(silent_score.per_valid_scalar) > 1e5 * correct_nll
    assert silent_diagnostics["scale_suspect"] is True

    # (c) the reverse mix (DN numbers against a normalised-domain query) is identifiable too
    reverse = model.score(
        correct, torch.full((3,), 2048.0 / CFA_SCALE, dtype=DTYPE), units=CFA_UNIT
    )
    assert float(reverse.per_valid_scalar) > 1e5 * correct_nll
    assert (
        model.diagnostics(
            correct, torch.full((3,), 2048.0 / CFA_SCALE, dtype=DTYPE), units=CFA_UNIT
        )["scale_suspect"]
        is True
    )


def test_scale_mismatch_keeps_finite_float32_gradients():
    """The reported symptom: a far-off (wrong-domain) float32 score must not give NaN grads."""
    model = _dn_sensor_model()
    values = torch.tensor([2046.0, 2050.0, 2048.0], dtype=torch.float32)
    for mean_value, variance_value in (
        (2048.0, 4.0),
        (2048.0 / CFA_SCALE, 4.0 / CFA_SCALE**2),
        (2048.0 * CFA_SCALE, 4.0 * CFA_SCALE**2),
    ):
        mean = torch.full((3,), mean_value, dtype=torch.float32, requires_grad=True)
        prediction = RawPrediction(
            mean=mean,
            variance=torch.full((3,), variance_value, dtype=torch.float32),
            masks={},
            units=CFA_UNIT,
        )
        score = model.score(prediction, values, units=CFA_UNIT)
        score.raw_nll_or_energy.backward()
        assert torch.isfinite(score.raw_nll_or_energy), mean_value
        assert bool(torch.isfinite(mean.grad).all()), (mean_value, mean.grad)


def test_synthetic_cfa_declared_variance_is_reproduced_in_dn_units():
    """The DN-unit declaration matches the generator's declared noise variance exactly."""
    measurement = _cfa_flat_measurement()
    plane = synth.cfa_pixel_plane_index("RGGB", 8, 8)
    response = np.asarray([0.95, 1.0, 1.0, 0.75], dtype=np.float64)
    mean_signal = response[plane] * CFA_BACKGROUND
    mean_dn = CFA_BLACK + mean_signal * CFA_SCALE
    variance_normalised = mean_signal / CFA_SHOT_NORMALISED + CFA_READ_NORMALISED**2
    variance_dn = variance_normalised * CFA_SCALE**2

    model = _dn_sensor_model(shot_noise_scale=CFA_SHOT_UNITS, read_noise_signal=CFA_READ_UNITS)
    signal_dn = torch.tensor(mean_dn - CFA_BLACK, dtype=DTYPE)
    assert torch.allclose(
        model.sensor_variance(signal_dn), torch.tensor(variance_dn, dtype=DTYPE), rtol=1e-12
    )
    prediction = RawPrediction(
        mean=torch.tensor(mean_dn, dtype=DTYPE),
        variance=model.sensor_variance(signal_dn),
        masks={"valid": torch.as_tensor(np.asarray(measurement.valid_mask))},
        units=measurement.unit,
    )
    score = model.score(prediction, measurement)
    assert 0.0 < float(score.per_valid_scalar) < 20.0
    assert float(score.local_stats["n_valid_scalars"]) == 64.0
    assert model.diagnostics(prediction, measurement)["scale_suspect"] is False


def _log_interval_midpoint(lower: float, upper: float, *, points: int = 20001) -> float:
    """Independent log-space midpoint quadrature of the Gaussian interval probability."""
    width = (upper - lower) / points
    logs = []
    for index in range(points):
        t = lower + (index + 0.5) * width
        logs.append(-0.5 * t * t - 0.5 * math.log(2.0 * math.pi))
    peak = max(logs)
    total = sum(math.exp(value - peak) for value in logs)
    return peak + math.log(total * width)


def test_interval_probability_matches_log_space_quadrature_in_deep_tails():
    for centre, sigma in ((19.5, 1.0), (19.5, 3.0), (30.0, 1.0), (-19.5, 1.0), (-45.0, 1.0)):
        got = sensor_mod.quantisation_log_prob(
            torch.tensor([centre], dtype=DTYPE),
            torch.zeros(1, dtype=DTYPE),
            torch.tensor([sigma * sigma], dtype=DTYPE),
            step=1.0,
        )
        reference = _log_interval_midpoint((centre - 0.5) / sigma, (centre + 0.5) / sigma)
        assert float(got[0]) == pytest.approx(reference, rel=1e-6, abs=1e-9), (centre, sigma)


def test_deep_tail_quantisation_keeps_finite_float32_gradients():
    for value in (-89557.0, -1000.0, -50.0, 0.0, 50.0, 1000.0, 89557.0):
        mean = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
        log_prob = sensor_mod.quantisation_log_prob(
            torch.tensor([value], dtype=torch.float32), mean, torch.tensor([1.0]), step=1.0
        )
        log_prob.backward()
        assert torch.isfinite(log_prob).all(), value
        assert bool(torch.isfinite(mean.grad).all()), (value, mean.grad)

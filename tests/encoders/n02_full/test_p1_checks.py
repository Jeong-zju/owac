"""Absolute-check tests for the N02 P1 checks module (T2).

The regression these tests exist for is the old failure mode: two local bright spots with
large standardised residuals were accepted because the whole-image mean stayed normal.
The check must keep the whole-image mean **and** the local window maximum, report both, and
call such a sequence ``model_inadequate``.  Calibration and test must use the identical
viewing rule (enforced by hash), the threshold is a frozen empirical 95% quantile, and the
normal-sequence false-positive rate must stay at or below 5%.
"""

from __future__ import annotations

import dataclasses
import math
import random

import pytest
import torch

from owac.frameworks.n02_full.checks import (
    STATUSES,
    CheckVerdict,
    ResidualView,
    ViewingRule,
    ViewingRuleMismatchError,
    WindowedCalibrator,
    sequence_statistics,
    windowed_statistics,
)

DTYPE = torch.float64

SIZE = 16
WINDOW = 4
CALIBRATION_SEED = 20261014
CALIBRATION_SEQUENCES = 400
TEST_SEED = 202
TEST_SEQUENCES = 800

BIG_SIZE = 24
BIG_CALIBRATION_SEED = 99
BIG_CALIBRATION_SEQUENCES = 200


def normal_map(rng: random.Random, size: int = SIZE) -> torch.Tensor:
    """Deterministic standard-normal map (stdlib PRNG and Box-Muller, version stable)."""
    values = []
    for _ in range(size * size):
        first = max(rng.random(), 1e-12)
        second = rng.random()
        values.append(math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second))
    return torch.tensor(values, dtype=DTYPE).reshape(size, size)


def normal_group(rng: random.Random, size: int = SIZE) -> list[ResidualView]:
    return [ResidualView(normal_map(rng, size=size), scale="stride8", frame_index=0)]


def view(residual: torch.Tensor, *, scale: str = "stride8", frame: int = 0) -> ResidualView:
    return ResidualView(residual, scale=scale, frame_index=frame, region_id="full")


@pytest.fixture(scope="module")
def rule() -> ViewingRule:
    return ViewingRule(scales=("stride8",), frames=(0,), window=WINDOW)


@pytest.fixture(scope="module")
def calibration(rule: ViewingRule):
    rng = random.Random(CALIBRATION_SEED)
    statistics = sequence_statistics(
        [normal_group(rng) for _ in range(CALIBRATION_SEQUENCES)], rule=rule
    )
    return WindowedCalibrator.fit(statistics, rule_hash=rule.rule_hash()), statistics


@pytest.fixture(scope="module")
def big_window_calibration():
    rule = ViewingRule(scales=("stride8",), frames=(0,), window=WINDOW)
    rng = random.Random(BIG_CALIBRATION_SEED)
    statistics = sequence_statistics(
        [normal_group(rng, size=BIG_SIZE) for _ in range(BIG_CALIBRATION_SEQUENCES)], rule=rule
    )
    return rule, WindowedCalibrator.fit(statistics, rule_hash=rule.rule_hash())


def test_check_verdict_has_exactly_the_frozen_fields_and_three_statuses():
    assert [item.name for item in dataclasses.fields(CheckVerdict)] == [
        "status",
        "interference_explainable",
        "local_residual_positions",
        "calibration_version",
        "statistics",
        "viewing_rule_hash",
    ]
    assert STATUSES == ("compatible", "insufficient_information", "model_inadequate")
    for status in STATUSES:
        verdict = CheckVerdict(status, False, (), "v", {}, "hash")
        assert verdict.status == status
    with pytest.raises(ValueError):
        # an anomaly never names a specific relation: only these three statuses exist
        CheckVerdict("relation_x_is_wrong", False, (), "v", {}, "hash")


def test_rule_hash_covers_scales_frames_regions_window_and_statistic():
    base = ViewingRule(scales=("stride8",), frames=(0,), window=4)
    assert base.rule_hash() == ViewingRule(scales=("stride8",), frames=(0,), window=4).rule_hash()
    assert len(base.rule_hash()) == 64
    variants = [
        ViewingRule(scales=("stride16",), frames=(0,), window=4),
        ViewingRule(scales=("stride8",), frames=(1,), window=4),
        ViewingRule(scales=("stride8",), frames=(0,), regions=("center",), window=4),
        ViewingRule(scales=("stride8",), frames=(0,), window=5),
        ViewingRule(scales=("stride8",), frames=(0,), window=4, use_local_maximum=False),
        ViewingRule(scales=("stride8",), frames=(0,), window=4, min_valid_samples=10),
    ]
    for variant in variants:
        assert variant.rule_hash() != base.rule_hash(), variant.as_dict()


def test_calibrator_freezes_the_empirical_quantile_and_never_re_estimates():
    statistics = [float(index) for index in range(100)]
    calibrator = WindowedCalibrator.fit(statistics, quantile=0.95, rule_hash="rule-a")
    ordered = sorted(statistics)
    assert calibrator.threshold() == ordered[math.ceil(0.95 * (len(ordered) - 1))]
    assert calibrator.calibration_sample_count == 100
    assert calibrator.quantile == 0.95
    assert calibrator.viewing_rule_hash == "rule-a"
    # non-finite calibration values are dropped, and an all-nonfinite window is refused
    filtered = WindowedCalibrator.fit([float("nan"), *statistics], rule_hash="rule-a")
    assert filtered.threshold() == calibrator.threshold()
    with pytest.raises(ValueError):
        WindowedCalibrator.fit([float("nan"), float("-inf")], rule_hash="rule-a")
    with pytest.raises(ValueError):
        WindowedCalibrator.fit(statistics, quantile=0.0, rule_hash="rule-a")


def test_calibrated_threshold_matches_the_sequence_quantile(calibration):
    calibrator, statistics = calibration
    ordered = sorted(statistics)
    assert calibrator.threshold() == ordered[math.ceil(0.95 * (len(ordered) - 1))]
    assert calibrator.calibration_sample_count == CALIBRATION_SEQUENCES


def test_assess_scalar_contract_reports_all_three_statuses(calibration):
    calibrator, _ = calibration
    low = calibrator.assess([0.0], version="dev-1")
    assert low.status == "compatible"
    assert low.calibration_version == "dev-1"
    assert low.viewing_rule_hash == calibrator.viewing_rule_hash
    assert low.statistics["threshold"] == calibrator.threshold()
    high = calibrator.assess([calibrator.threshold() + 1.0], version="dev-1")
    assert high.status == "model_inadequate"
    empty = calibrator.assess([], version="dev-1")
    assert empty.status == "insufficient_information"
    assert math.isnan(empty.statistics["max_sequence_statistic"])


def test_viewing_rule_mismatch_and_missing_views_are_refused(calibration, rule):
    calibrator, _ = calibration
    frame_view = view(torch.zeros(SIZE, SIZE, dtype=DTYPE))
    with pytest.raises(ViewingRuleMismatchError):
        calibrator.assess_views(
            [frame_view],
            version="dev-1",
            rule=ViewingRule(scales=("stride16",), frames=(0,), window=WINDOW),
        )
    two_frame_rule = ViewingRule(scales=("stride8",), frames=(0, 1), window=WINDOW)
    two_frame_calibrator = WindowedCalibrator.fit(
        [0.0, 1.0, 2.0], rule_hash=two_frame_rule.rule_hash()
    )
    with pytest.raises(ViewingRuleMismatchError):
        two_frame_calibrator.assess_views([frame_view], version="dev-1", rule=two_frame_rule)


def test_windowed_statistics_report_whole_image_mean_and_local_maximum(rule):
    residual = torch.zeros(SIZE, SIZE, dtype=DTYPE)
    residual[0:4, 0:4] = 5.0
    statistics = windowed_statistics(view(residual), rule=rule)
    assert math.isclose(statistics["global_mean_abs_standardised_residual"], 0.3125, rel_tol=1e-12)
    assert math.isclose(statistics["local_max_abs_standardised_residual"], 5.0, rel_tol=1e-12)
    assert math.isclose(statistics["statistic"], 5.0, rel_tol=1e-12)
    assert statistics["n_valid_samples"] == float(SIZE * SIZE)


def test_only_fully_valid_windows_contribute_a_local_maximum(rule):
    residual = torch.zeros(SIZE, SIZE, dtype=DTYPE)
    valid = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    valid[0:2, 0:2] = True
    residual[0:2, 0:2] = 10.0
    statistics = windowed_statistics(ResidualView(residual, valid=valid), rule=rule)
    assert statistics["n_valid_samples"] == 4.0
    assert statistics["n_full_windows"] == 0.0
    assert statistics["local_max_abs_standardised_residual"] == float("-inf")
    assert math.isclose(statistics["statistic"], 10.0, rel_tol=1e-12)


def test_insufficient_information_when_no_valid_samples(calibration, rule):
    calibrator, _ = calibration
    invalid = ResidualView(
        torch.zeros(SIZE, SIZE, dtype=DTYPE),
        valid=torch.zeros(SIZE, SIZE, dtype=torch.bool),
        scale="stride8",
        frame_index=0,
    )
    verdict = calibrator.assess_views([invalid], version="dev-1", rule=rule)
    assert verdict.status == "insufficient_information"
    assert not verdict.interference_explainable


def test_insufficient_information_when_a_declared_view_is_below_the_minimum():
    rule = ViewingRule(
        scales=("stride8",), frames=(0,), window=WINDOW, min_valid_samples=SIZE * SIZE + 1
    )
    calibrator = WindowedCalibrator.fit([0.0, 1.0, 2.0], rule_hash=rule.rule_hash())
    verdict = calibrator.assess_views(normal_group(random.Random(5)), version="dev-min", rule=rule)
    assert verdict.status == "insufficient_information"
    assert not verdict.interference_explainable


def test_double_bright_spots_are_model_inadequate_not_averaged_away(big_window_calibration):
    rule, calibrator = big_window_calibration
    residual = normal_map(random.Random(1234), size=BIG_SIZE)
    residual[3:6, 3:6] = 6.0
    residual[10:13, 10:13] = 5.5
    verdict = calibrator.assess_views([view(residual)], version="dev-bright", rule=rule)
    assert verdict.status == "model_inadequate", verdict.statistics
    statistics = verdict.statistics
    # both aggregates are reported, and the whole-image mean alone would not have fired
    assert (
        statistics["global_mean_abs_standardised_residual"]
        < statistics["threshold"]
        < statistics["local_max_abs_standardised_residual"]
    )
    assert len(verdict.local_residual_positions) >= 2
    spots = [(4, 4), (12, 12)]
    for row, column in verdict.local_residual_positions:
        distance = min(max(abs(row - y), abs(column - x)) for y, x in spots)
        assert distance <= 3, (row, column)
    assert verdict.calibration_version == "dev-bright"
    assert verdict.viewing_rule_hash == rule.rule_hash()


def test_a_uniform_elevated_residual_is_model_inadequate_through_the_mean(
    big_window_calibration,
):
    rule, calibrator = big_window_calibration
    residual = normal_map(random.Random(21), size=BIG_SIZE) + 3.0
    verdict = calibrator.assess_views([view(residual)], version="dev-mean", rule=rule)
    assert verdict.status == "model_inadequate"
    assert (
        verdict.statistics["global_mean_abs_standardised_residual"]
        > verdict.statistics["threshold"]
    )


def test_a_normal_sequence_is_compatible_and_reports_both_aggregates(calibration, rule):
    calibrator, _ = calibration
    verdict = calibrator.assess_views(
        normal_group(random.Random(TEST_SEED + 1)), version="dev-normal", rule=rule
    )
    assert verdict.status == "compatible"
    assert not verdict.interference_explainable
    assert verdict.local_residual_positions == ()
    assert "global_mean_abs_standardised_residual" in verdict.statistics
    assert "local_max_abs_standardised_residual" in verdict.statistics


def test_declared_interference_can_explain_an_otherwise_anomalous_sequence(
    big_window_calibration,
):
    rule, calibrator = big_window_calibration
    raw = normal_map(random.Random(7), size=BIG_SIZE) + 4.0
    corrected = raw - 4.0
    unexplained = calibrator.assess_views([view(raw)], version="dev-eta", rule=rule)
    assert unexplained.status == "model_inadequate"
    assert not unexplained.interference_explainable
    explained = calibrator.assess_views(
        [view(raw)],
        version="dev-eta",
        rule=rule,
        interference_views=[view(corrected)],
    )
    assert explained.status == "compatible"
    assert explained.interference_explainable
    assert explained.statistics["interference_statistic"] <= explained.statistics["threshold"]


def test_empirical_false_positive_rate_stays_at_or_below_five_percent(calibration, rule):
    calibrator, _ = calibration
    rng = random.Random(TEST_SEED)
    statistics = sequence_statistics([normal_group(rng) for _ in range(TEST_SEQUENCES)], rule=rule)
    threshold = calibrator.threshold()
    false_positives = sum(1 for value in statistics if value > threshold)
    limit = int(0.05 * TEST_SEQUENCES)
    message = (
        f"false positives {false_positives}/{TEST_SEQUENCES} "
        f"({false_positives / TEST_SEQUENCES:.4f}) against a frozen threshold of {threshold:.6f}"
    )
    assert false_positives >= 1, f"calibrator never fires: {message}"
    assert false_positives <= limit, message
    assert false_positives / TEST_SEQUENCES <= 0.05, message

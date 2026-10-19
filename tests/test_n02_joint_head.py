"""Single-factor and frozen-value contracts for the joint-head control."""

import copy

import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_joint_head import (  # noqa: E402
    false_score_summary,
    fit_arm,
    fixed_coefficient,
    gradient_profile,
    joint_components,
    joint_criteria,
    weighted_objective,
)
from owac.evaluation.n02_measurement_isolation import (  # noqa: E402
    FixedGeometryHead,
    head_objective,
    state_digest,
    tensor_digest,
)
from owac.frameworks.n02.measurement import MeasurementHead  # noqa: E402


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    source = MeasurementHead(8, 4, 3, head_kind="geometry_coupled").double()
    with torch.no_grad():
        source.contrast.weight.normal_(std=0.1)
    cache = {
        "fields": torch.randn(4, 4, 8, 4, 4, dtype=torch.float64),
        "templates": torch.randn(4, 2, 3, 4, 4, dtype=torch.float64) * 0.1,
        "total_variance": torch.rand(4, 3, 4, 4, dtype=torch.float64) + 0.1,
        "observed": torch.randn(4, 3, 4, 4, dtype=torch.float64) * 0.1,
        "identities": [{"seed": 1, "world": str(index % 2)} for index in range(4)],
    }
    cache["templates"][:2, 1] = cache["templates"][:2, 0]
    return source, cache


def test_lambda_one_matches_original_paired_nll_value_and_all_parameter_gradients():
    source, cache = fixture()
    head = FixedGeometryHead(source)
    parameters = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    old = head_objective(head, cache, [0, 1, 2, 3], "paired_nll", 1)
    old_gradients = torch.autograd.grad(old, parameters)
    decomposed = weighted_objective(joint_components(head, cache), 1.0)
    new_gradients = torch.autograd.grad(decomposed, parameters)
    torch.testing.assert_close(decomposed, old, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(new_gradients, old_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_lambda_uses_only_full_train_initial_component_values():
    source, cache = fixture()
    head = FixedGeometryHead(source)
    original = state_digest(head)
    coefficient, losses = fixed_coefficient(head, cache)
    assert coefficient * losses["contrast"] == pytest.approx(losses["common"])
    assert state_digest(head) == original
    with torch.no_grad():
        prediction = head(cache["fields"])
    invalid = copy.copy(cache)
    invalid["templates"] = torch.stack([prediction[:, 0], prediction[:, 1]], 1)
    with pytest.raises(ValueError, match="positive and finite"):
        fixed_coefficient(head, invalid)


def test_shared_trunk_gradient_probe_has_hand_checkable_conflict():
    class ToyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Module()
            self.head.trunk = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.head.trunk.weight)

        def forward(self, fields):
            value = self.head.trunk(fields).reshape(-1, 1, 1, 1)
            # c=d=w: even mean0, odd mean2w. At w0, target c1,d-1
            # gives dLc/dw=-1, dLd/dw=+1.
            return torch.stack([torch.zeros_like(value), 2 * value], 1)

    cache = {
        "fields": torch.ones(1, 1),
        "templates": torch.tensor([[[[[2.0]]], [[[0.0]]]]]),
        "total_variance": torch.ones(1, 1, 1, 1),
    }
    report = gradient_profile(ToyHead(), cache, 3)
    assert report["common_gradient_norm"] == pytest.approx(1)
    assert report["contrast_gradient_norm"] == pytest.approx(1)
    assert report["cosine"] == pytest.approx(-1)
    assert report["common_plus_contrast_gradient_norm"] == pytest.approx(0)
    assert report["common_plus_weighted_contrast_gradient_norm"] == pytest.approx(2)


def test_no_cue_score_reports_false_confidence_from_common_bias():
    cache = {
        "templates": torch.zeros(2, 2, 1, 1, 1),
        "observed": torch.zeros(2, 1, 1, 1),
        "total_variance": torch.ones(2, 1, 1, 1),
        "identities": [{"seed": 1, "world": "no_cue"}] * 2,
    }
    prediction = torch.tensor([[[[[0.0]]], [[[2.0]]]]]).expand(2, -1, -1, -1, -1)
    report = false_score_summary(prediction, cache)
    assert report["frame_mean_abs_log_bayes"] == pytest.approx(2)
    assert report["sequence_false_confidence_ge_0_95"] == 1
    corrected_common = prediction - 1
    corrected = false_score_summary(corrected_common, cache)
    assert corrected["frame_mean_abs_log_bayes"] == 0
    assert corrected["sequence_mean_abs_p1_drift"] == 0


def test_both_fit_and_heldout_common_error_must_not_degrade():
    initial = {
        "train": {"common_weighted_rmse": 0.1},
        "evaluation": {"common_weighted_rmse": 0.2},
    }
    result = copy.deepcopy(initial)
    result["evaluation"].update(
        contrast_nmse=0.2,
        cosine=0.8,
        actual_head_choices={"both_correct": 12, "total": 16},
        no_cue_false_contrast_ratio=0.05,
    )
    assert all(joint_criteria(result, initial).values())
    result["train"]["common_weighted_rmse"] += 1e-5
    assert not joint_criteria(result, initial)["train_common_error_nondegrading"]
    assert joint_criteria(result, initial)["evaluation_common_error_nondegrading"]


def test_joint_optimization_preserves_cached_values_and_original_parameters(tmp_path):
    source, cache = fixture()
    original = state_digest(source)
    hashes = {key: tensor_digest(value) for key, value in cache.items() if torch.is_tensor(value)}
    coefficient, _ = fixed_coefficient(FixedGeometryHead(source), cache)
    reports = fit_arm(
        source,
        cache,
        cache,
        coefficient=coefficient,
        schedule=[0, 1],
        signal_scale=0.01,
        output=tmp_path,
        label="test",
    )
    assert set(reports) == {"0", "1", "2"}
    for snapshot in reports.values():
        for split in ("train", "evaluation"):
            assert set(snapshot[split]["components"]) == {"common", "contrast", "log_variance"}
    assert state_digest(source) == original
    assert hashes == {
        key: tensor_digest(value) for key, value in cache.items() if torch.is_tensor(value)
    }
    assert all(parameter.grad is None for parameter in source.parameters())

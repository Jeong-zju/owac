"""Information-path and physical-metric checks for the frozen-head diagnostic."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.evaluation.n02_measurement_isolation import (  # noqa: E402
    FixedGeometryHead,
    class_predictions,
    contrast_metrics,
    frame_cache,
    head_objective,
    optimize_head,
    pair_choices,
    state_digest,
    tensor_digest,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.measurement import MeasurementHead  # noqa: E402
from owac.frameworks.n02.model import N02Encoder  # noqa: E402
from owac.training.n02 import _frame, synthetic_batch  # noqa: E402


def tiny_head():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    return MeasurementHead(8, 4, 3, head_kind="geometry_coupled")


def test_fixed_geometry_is_production_intervention_without_target_input():
    source = tiny_head()
    with torch.no_grad():
        source.contrast.bias.fill_(0.25)
    probe = FixedGeometryHead(source)
    fields = torch.randn(2, 4, 8, 4, 4)
    predicted = probe(fields)
    _, _, geometry = probe.head(fields)
    assert torch.equal(geometry[..., 0], torch.tensor([[-1.0, 1.0, -1.0, 1.0]]).expand(2, -1))
    torch.testing.assert_close(predicted[:, 1] - predicted[:, 0], torch.full((2, 3, 4, 4), 0.5))
    with pytest.raises(ValueError, match="detached"):
        probe(fields.requires_grad_())


def test_physical_contrast_metrics_distinguish_gain_sign_and_no_cue_hallucination():
    template = torch.zeros(3, 2, 1, 2, 2)
    template[:2, 1, :, 0, 0] = 2
    exact = contrast_metrics(template, template, 0.25)
    assert exact["contrast_nmse"] == 0
    assert exact["cosine"] == pytest.approx(1)
    assert exact["projection_gain"] == pytest.approx(1)
    assert exact["proper_vs_swapped_fraction"] == 1
    assert exact["no_cue_false_contrast_ratio"] == 0
    swapped = contrast_metrics(template.flip(1), template, 0.25)
    assert swapped["projection_gain"] == -1
    assert swapped["cosine"] == pytest.approx(-1)
    assert swapped["contrast_nmse"] == 4
    assert swapped["proper_vs_swapped_fraction"] == 0
    hallucination = template.clone()
    hallucination[2, 1].fill_(1)
    assert contrast_metrics(hallucination, template, 0.25)["no_cue_false_contrast_ratio"] == 1


def test_paired_score_requires_both_worlds_to_choose_their_own_template():
    target = torch.tensor([[[[[0.0]]], [[[1.0]]]]]).expand(2, -1, -1, -1, -1)
    cache = {
        "observed": torch.tensor([[[[0.0]]], [[[1.0]]]]),
        "total_variance": torch.full((2, 1, 1, 1), 0.1),
        "identities": [{"seed": 1, "world": world} for world in ("0", "1")],
    }
    assert pair_choices(target, cache)["both_correct"] == 1
    assert pair_choices(target.flip(1), cache)["both_correct"] == 0
    assert pair_choices(torch.zeros_like(target), cache)["both_correct"] == 0


@pytest.mark.parametrize("objective", ["paired_nll", "contrast_mse"])
def test_head_fit_cannot_update_encoder_or_cached_variance(objective):
    source = tiny_head()
    original = state_digest(source)
    fields = torch.randn(4, 4, 8, 4, 4)
    with torch.no_grad():
        targets = class_predictions(FixedGeometryHead(source)(fields))
        targets[:, 0] -= 0.05
        targets[:, 1] += 0.05
    cache = {
        "fields": fields,
        "templates": targets,
        "total_variance": torch.rand(4, 3, 4, 4) + 0.1,
    }
    variance_before = tensor_digest(cache["total_variance"])
    head, _ = optimize_head(
        source,
        cache,
        objective=objective,
        schedule=[0, 1],
        signal_scale=0.0025,
    )
    assert state_digest(source) == original
    assert tensor_digest(cache["total_variance"]) == variance_before
    assert all(parameter.grad is None for parameter in source.parameters())
    assert not torch.equal(head.head.contrast.bias, source.contrast.bias)
    for name in ("geometry", "log_variance"):
        for learned, initial in zip(
            getattr(head.head, name).parameters(), getattr(source, name).parameters(), strict=True
        ):
            assert torch.equal(learned, initial)
            assert learned.grad is None
    reference_output = head(fields).detach()
    changed_targets = copy.copy(cache)
    changed_targets["templates"] = targets + 100
    assert head_objective(head, cache, [0, 1], "paired_nll", 1) != head_objective(
        head, changed_targets, [0, 1], "paired_nll", 1
    )
    assert torch.equal(head(fields).detach(), reference_output)


def test_template_targets_do_not_enter_causal_encoder_cache():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = N02Config.from_json("configs/encoders/n02/tiny_contracts.json")
    model = N02Encoder(cfg).eval()
    batch = synthetic_batch(
        cfg,
        np.random.default_rng(9),
        batch_size=2,
        num_frames=4,
        frame_size=64,
        seed=9,
        split="train",
    )
    args = (model, _frame(batch, 0), batch.contexts[0], None)
    original, _ = frame_cache(
        *args, batch.counterfactual_templates[0], [{"element": i} for i in range(2)], [False] * 2
    )
    changed, _ = frame_cache(
        *args,
        batch.counterfactual_templates[0] + 100,
        [{"element": i} for i in range(2)],
        [True] * 2,
    )
    for key in ("fields", "observed", "total_variance", "model_variance", "sensor_variance"):
        assert torch.equal(original[key], changed[key])
    assert not torch.equal(original["templates"], changed["templates"])
    assert not original["fields"].requires_grad

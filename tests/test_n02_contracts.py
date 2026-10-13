"""Observable numerical/causal contracts for the optional N02 research variant."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from owac.core.encoder_contract import (  # noqa: E402
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.measurement import (  # noqa: E402
    MeasurementHead,
    MeasurementPrediction,
    compute_measurement_energy,
)
from owac.frameworks.n02.model import N02Encoder  # noqa: E402
from owac.frameworks.n02.revision import RevisionController  # noqa: E402
from owac.training.n02 import (  # noqa: E402
    _frame,
    counterfactual_measurement_loss,
    synthetic_batch,
)


def config():
    return N02Config.from_json("configs/encoders/n02/tiny_contracts.json")


def evidence(means, *, observed=None, valid=None, variance=0.04, model_variance=0.0, geometry=None):
    means = torch.as_tensor(means, dtype=torch.float64)
    if means.ndim == 1:
        means = means[:, None, None, None]
    K, channels, height, width = means.shape
    if geometry is None:
        geometry = torch.arange(K) % 2 * 2.0 - 1
    prediction = MeasurementPrediction(
        means[None, None],
        torch.full_like(means[None, None], model_variance),
        torch.as_tensor(geometry).reshape(1, 1, K, 1),
    )
    frame = SimpleNamespace(
        values=torch.zeros(1, channels, height, width, dtype=torch.float64)
        if observed is None
        else observed,
        valid_mask=torch.ones(1, channels, height, width, dtype=torch.bool)
        if valid is None
        else valid,
        saturated_mask=None,
        bad_mask=None,
        variance=torch.full((1, channels, height, width), variance, dtype=torch.float64),
        metadata={"variance_is_estimate": False},
    )
    result = compute_measurement_energy(
        prediction, SimpleNamespace(frames=[frame]), config().measurement, view_index=0
    )
    return result


def test_scoring_units_padding_and_total_variance():
    # Total variance .04+.21=.25. One scalar y=0 discriminates means 0/1
    # by exactly two log-odds units, irrespective of three irrelevant positions.
    result = evidence([0.0, 1.0, 0.0, 1.0], model_variance=0.21)
    assert float(result.energy[0, 1] - result.energy[0, 0]) == pytest.approx(2)
    means = torch.zeros(4, 1, 1, 4)
    means[1::2, 0, 0, 0] = 1
    padded = evidence(means, model_variance=0.21)
    assert float(padded.energy[0, 1] - padded.energy[0, 0]) == pytest.approx(2)
    prior = torch.tensor([0.025, 0.475, 0.025, 0.475], dtype=torch.float64)
    p = torch.softmax(prior.log() - result.energy[0], 0)[::2].sum()
    assert float(p) == pytest.approx(1 / (1 + 19 * np.exp(-2)))


@pytest.mark.parametrize("case", ["equal", "same_geometry", "all_bad", "missing", "uncertain"])
def test_prediction_error_is_not_automatically_semantic_refutation(case):
    kwargs = {}
    means = [0.0, 1.0, 0.0, 1.0]
    if case == "equal":
        means = [1.0] * 4
    elif case == "same_geometry":
        kwargs["geometry"] = [-1.0] * 4
    elif case == "all_bad":
        means = [2.0, 4.0, 2.0, 4.0]
    elif case == "missing":
        kwargs["valid"] = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    elif case == "uncertain":
        kwargs["model_variance"] = 1.0
    result = evidence(means, **kwargs)
    controller = RevisionController(config())
    mask = controller.semantic_map(result)
    assert not mask.any()
    penalty = controller.withdrawal_energy(
        result.normalized_residual[:, None],
        result.support_mask[:, None],
        temperature=1,
        semantic_refutation=mask,
    )
    assert torch.equal(penalty, torch.zeros_like(penalty))


def test_real_refutation_is_local_and_geometry_specific():
    means = torch.zeros(4, 1, 3, 3)
    means[1::2, 0, 1, 1] = 1
    controller = RevisionController(config())
    result = evidence(means)
    mask = controller.semantic_map(result)
    expected = torch.zeros(1, 1, 4, 3, 3, dtype=torch.bool)
    expected[0, 0, 1::2, 1, 1] = True
    assert torch.equal(mask, expected)


def test_geometry_is_a_causal_input_to_the_returned_raw_prediction():
    torch.manual_seed(0)
    head = MeasurementHead(8, 4, 3, head_kind="geometry_coupled")
    fields = torch.randn(1, 4, 8, 4, 4)
    mean, _, _ = head(fields)
    # Initial contrast is zero: no random per-slot reconstruction advantage.
    assert torch.equal(mean[:, 0], mean[:, 1])
    with torch.no_grad():
        head.contrast.bias.fill_(0.1)
    positive, _, geo = head(fields)
    hook = head.geometry.register_forward_hook(lambda _m, _i, out: -out)
    negative, _, reversed_geo = head(fields)
    hook.remove()
    torch.testing.assert_close(reversed_geo, -geo)
    torch.testing.assert_close(positive - negative, 0.2 * geo[..., :1, None, None].expand_as(mean))
    assert float((positive - negative).detach().abs().max()) > 0


def test_template_supervision_has_gradients_but_no_encoder_input_route():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = config()
    model = N02Encoder(cfg)
    batch = synthetic_batch(
        cfg,
        np.random.default_rng(9),
        batch_size=1,
        num_frames=4,
        frame_size=64,
        seed=9,
        split="train",
    )
    first = model.step(_frame(batch, 0), batch.contexts[0], None, ComputeBudget(4, 4))
    loss = counterfactual_measurement_loss(
        model, first.representation, batch.counterfactual_templates[0], sensor_variance=0.01**2 / 16
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.measurement.head.contrast.weight.grad.abs().sum() > 0
    assert model.measurement.head.log_variance.weight.grad.abs().sum() > 0
    batch.counterfactual_templates[0].add_(100)
    with torch.no_grad():
        second = model.step(_frame(batch, 0), batch.contexts[0], None, ComputeBudget(4, 4))
    torch.testing.assert_close(
        first.representation.weights, second.representation.weights, rtol=0, atol=0
    )
    torch.testing.assert_close(first.representation.h, second.representation.h, rtol=0, atol=0)


def test_fit_updates_do_not_reset_or_withdraw_unidentifiable_hypotheses():
    torch.set_num_threads(1)
    model = N02Encoder(config())
    batch = synthetic_batch(
        config(),
        np.random.default_rng(3),
        batch_size=1,
        num_frames=4,
        frame_size=64,
        seed=3,
        split="train",
    )
    with torch.no_grad():
        first = model.step(_frame(batch, 0), batch.contexts[0], None, ComputeBudget(1, 4))
        state = first.state
        controller = model.reviser
        field = state.h
        residual = torch.full((1, 1, 4, 16, 16), 20.0)
        kwargs = dict(
            residual_feature=torch.ones_like(field),
            prior_feature=torch.ones_like(field),
            location_code=model._position_code(field),
            semantic_refutation=torch.zeros_like(residual, dtype=torch.bool),
        )
        # Use the encoder's actual feature shapes, not scalar stand-ins.
        kwargs["residual_feature"] = model._residual_feature(
            evidence(torch.zeros(4, 3, 16, 16)), field
        )
        kwargs["prior_feature"] = model._prior_feature(state, field)
        result = controller.step(state, residual, **kwargs)
        assert result.updated_fraction == 1
        assert not result.contradiction.any() and not result.reset.any()
        assert not torch.equal(result.h, field)
        controller.revision = replace(controller.revision, separate_fit_updates=False)
        unchanged = controller.step(state, residual, **kwargs)
        assert unchanged.updated_fraction <= result.updated_fraction


def test_new_scoring_and_withdrawal_count_each_measurement_once():
    torch.set_num_threads(1)
    cfg = config()
    cfg.revision.update_scale = 0
    cfg.revision.reset_contradicted = False
    model = N02Encoder(cfg)

    def fixed_prediction(self, h):
        mean = h.new_tensor([0.0, 0.2, 0.0, 0.2]).reshape(1, 1, 4, 1, 1, 1)
        mean = mean.expand(h.shape[0], 1, 4, 3, *h.shape[-2:])
        geo = h.new_zeros(h.shape[0], 1, 4, 8)
        geo[..., 0] = h.new_tensor([-1, 1, -1, 1])
        return mean, torch.full_like(mean, np.log(0.0025)), geo

    model.measurement.predict = MethodType(fixed_prediction, model.measurement)
    context = EmbodimentContext()

    def frame(index):
        return MeasurementFrame(
            values=torch.zeros(1, 3, 8, 8),
            frame_ids=(index,),
            episode_ids=("contract",),
            metadata={"variance": 0.0001},
            metadata_available={"variance": True},
        )

    with torch.no_grad():
        first = model.step(frame(0), context, None, ComputeBudget(4, 4))
        after = model.step(frame(1), context, first.state, ComputeBudget(4, 4))
        single = model.step(frame(1), context, first.state, ComputeBudget(1, 4))
        repeated = model.step(frame(1), context, after.state, ComputeBudget(4, 4))
    assert after.representation.revision_energy.max() > 0
    torch.testing.assert_close(after.state.logits, single.state.logits, rtol=0, atol=0)
    torch.testing.assert_close(after.state.logits, repeated.state.logits, rtol=0, atol=0)
    assert after.diagnostics["likelihood_observations"] == 1
    assert after.diagnostics["observation_history"] == 2


def test_extension_requires_behavior_on_two_seeds_and_no_regression():
    from copy import deepcopy

    from owac.evaluation.n02_contracts import extension_decision

    result = {
        "gates": {
            key: {"passed": value, "total": 16}
            for key, value in (
                ("correct", 12),
                ("misleading", 4),
                ("unidentifiable", 12),
                ("stale", 4),
            )
        },
        "counterfactual": {"both_correct": 8},
    }
    dev = {str(seed): {str(step): deepcopy(result) for step in (100, 200)} for seed in (0, 1, 2)}

    def verdict():
        return extension_decision(dev, contracts_passed=True, finite=True)["extend_to_800"]

    assert verdict()
    assert not extension_decision(dev, contracts_passed=False, finite=True)["extend_to_800"]
    assert not extension_decision(dev, contracts_passed=True, finite=False)["extend_to_800"]
    dev["0"]["200"]["counterfactual"]["both_correct"] = 0
    assert verdict()  # two other seeds still qualify
    dev["1"]["200"]["counterfactual"]["both_correct"] = 0
    assert not verdict()
    dev["1"]["200"]["counterfactual"]["both_correct"] = 8
    dev["0"]["100"]["gates"]["stale"]["passed"] = 5
    assert not verdict()  # passing threshold alone cannot hide regression

"""Sensor, posterior and local-refutation regressions for the six source findings."""

import math
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from owac.core.encoder_contract import ComputeBudget, EmbodimentContext, MeasurementFrame
from owac.frameworks.n02.config import MeasurementConfig, N02Config, RevisionConfig
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.synthetic import (
    SyntheticGenerator,
    SyntheticOptions,
    cfa_sample,
    values_for_task,
)
from owac.training.n02 import step_terms


def generator(mode="rawvla_rgb3", pattern="RGGB", **kwargs):
    return SyntheticGenerator(
        SyntheticOptions(height=64, width=64, input_mode=mode, cfa_pattern=pattern, **kwargs),
        seed=100,
        split="validation",
    )


@pytest.mark.parametrize(
    "mode,pattern",
    [("rawvla_rgb3", "RGGB")] + [("bayer_cfa", p) for p in ("RGGB", "BGGR", "GRBG", "GBRG")],
)
def test_final_photosites_preserve_cue_and_overcome_wrong_95_percent_prior(mode, pattern):
    gen = generator(mode, pattern, noise_sigma=0.01)
    episode = gen.generate(seed=100, identifiable=True)
    clean = gen.clean_measurements(episode)
    # Independent optical law, explicitly sampled for each CFA phase/frame.
    for step in range(episode.num_frames):
        optical = 1 - episode.occluder_mask * (1 - episode.transmission[step])
        optical = cfa_sample(optical, pattern)[0] if mode == "bayer_cfa" else optical[None]
        np.testing.assert_allclose(
            episode.frames[step], optical * clean[step] + episode.noise[step], atol=1e-7
        )
    quality = gen.measurement_quality(episode)
    assert quality["identifiable_now"][-1]
    # Independent Gaussian likelihood ratio using actual emitted observations.
    logbf = 0.0
    for step in range(episode.num_frames):
        truth = values_for_task(episode.task).index(episode.value_over_time[step])
        templates = gen.measurement_templates(episode, step)
        errors = np.square(episode.frames[step] - templates).sum(axis=(1, 2, 3))
        logbf += (errors[1 - truth] - errors[truth]) / (2 * 0.01**2)
    assert logbf > math.log(19)
    assert 1 / (1 + math.exp(-min(logbf - math.log(19), 700))) > 0.99
    no_cue = gen.generate(seed=100, identifiable=False)
    assert not any(gen.measurement_quality(no_cue)["identifiable_now"])
    assert gen.infer_hypothesis(episode) == episode.value_over_time[-1]
    assert gen.verify_measurement(episode)["agrees"]
    # A renderer regression cannot be hidden by the pre-render cue flag.
    corrupted = replace(episode, frames=np.zeros_like(episode.frames))
    with pytest.raises(ValueError, match="final RAW"):
        gen.measurement_quality(corrupted)


def test_sensor_noise_variance_is_not_attenuated_and_strong_prior_does_not_change_raw():
    gen = generator(noise_sigma=0.01)
    other = SyntheticGenerator(
        replace(gen.options, prior_log_odds_max=1.6, strong_prior_probability=0),
        seed=100,
        split="validation",
    )
    episode = gen.generate(seed=100, identifiable=True)
    np.testing.assert_array_equal(
        episode.frames, other.generate(seed=100, identifiable=True).frames
    )
    # Compare emitted read noise in covered/uncovered areas over many photosites.
    for covered in (episode.occluder_mask > 0.9, episode.occluder_mask < 0.1):
        actual_variance = float(episode.noise[:, :, covered].var())
        assert actual_variance == pytest.approx(episode.metadata["noise_variance"], rel=0.12)
    priors = [gen.generate(seed=s).prior.log_odds for s in range(100, 164)]
    assert any(x == math.log(19) for x in priors)
    assert max(priors) > math.log(19)
    assert min(priors) < 1.6


def terms(energy, weights, identifiable=False, prior=None, anchor=0):
    rep = SimpleNamespace(
        measurement_energy=energy[:, None],
        weights=weights[:, None],
        geometry=torch.zeros(weights.shape[0], 1, weights.shape[1], 8),
    )
    return step_terms(
        rep,
        anchor=torch.tensor([anchor]),
        wrong=torch.tensor([1 - anchor]),
        post_contradiction=torch.tensor([identifiable]),
        observations=1,
        diversity_active=False,
        identifiable_now=torch.tensor([identifiable]),
        ambiguity_prior=prior,
    )


def test_actual_ambiguity_posterior_is_supervised_and_informative_prior_is_preserved():
    energy = torch.zeros(1, 4, requires_grad=True)
    bad = torch.tensor([[0.01, 0.49, 0.01, 0.49]], requires_grad=True)
    uniform = torch.full((1, 4), 0.25, requires_grad=True)
    bad_loss = terms(energy, bad)["mixture"]
    assert bad_loss > terms(energy, uniform)["mixture"]
    bad_loss.backward()
    assert bad.grad is not None and bad.grad[0, 0] < bad.grad[0, 1]
    # The posterior term has no fake dependency on measurement-only logits.
    assert energy.grad is None
    strong = torch.tensor([[0.475, 0.025, 0.475, 0.025]])
    good = terms(energy, strong, prior=strong)["mixture"]
    wrong = terms(energy, uniform, prior=strong)["mixture"]
    assert good < wrong


@pytest.mark.parametrize("K", [2, 4, 8])
@pytest.mark.parametrize("visible", [False, True])
def test_all_compatible_slots_receive_equal_measurement_gradient(K, visible):
    energy = torch.arange(K, dtype=torch.float32).reshape(1, K).requires_grad_()
    weights = torch.full((1, K), 1 / K)
    loss = terms(energy, weights, identifiable=visible)["measurement"]
    loss.backward()
    wanted = torch.ones(K) / K
    if visible:
        wanted[1::2] = 0
        wanted[::2] = 2 / K
    torch.testing.assert_close(energy.grad[0], wanted)
    if K >= 4:
        permutation = torch.arange(K)
        permutation[0], permutation[2] = 2, 0
        swapped = terms(energy[:, permutation], weights, identifiable=visible)["measurement"]
        torch.testing.assert_close(loss, swapped)


def model_config(**revision):
    return N02Config(
        K=2,
        D=8,
        stem_width=8,
        cycles=4,
        temperature=1,
        beta_prior=1,
        measurement=MeasurementConfig(robust="gaussian"),
        revision=RevisionConfig(**revision),
    )


def test_mixed_dependency_and_directed_two_hop_refutation_remain_local():
    model = N02Encoder(model_config(propagation_hops=2))
    source = torch.zeros(1, 1, 2, 5, 5, dtype=torch.bool)
    source[0, 0, 1, 1, 2] = True
    edge = torch.zeros(1, 1, 5, 5, 4, 2, 2)
    valid = torch.ones(1, 1, 5, 5, 4)
    # Target row mixes candidates: a weight less than one must still propagate.
    edge[0, 0, 2, 2, 0, 0] = torch.tensor([0.8, 0.2])
    edge[0, 0, 3, 2, 0, 0, 0] = 1
    selection, priority = model.reviser.propagate(source, edge, valid)
    assert int(selection.sum()) == 3
    assert priority[0, 0, 0, 2, 2] == pytest.approx(0.1)
    assert priority[0, 0, 0, 3, 2] == pytest.approx(0.05)
    # Unit chain decays once at each hop: 1, .5, .25.
    edge[0, 0, 2, 2, 0, 0] = torch.tensor([0.0, 1.0])
    _, priority = model.reviser.propagate(source, edge, valid)
    assert priority[0, 0, 0, 3, 2] == pytest.approx(0.25)
    valid[0, 0, 3, 2, 0] = 0
    selection, _ = model.reviser.propagate(source, edge, valid)
    assert not selection[0, 0, 0, 3, 2]


def test_negative_dependency_can_recover_without_changing_old_forward_semantics():
    model = N02Encoder(model_config())
    h = torch.zeros(1, 1, 2, 1, 3, 3)
    h[:, :, 1] = 2
    edge = torch.eye(2).expand(1, 1, 3, 3, 4, 2, 2).clone()
    edge[..., 0, 1] = -0.1
    edge.requires_grad_()
    valid = torch.ones(1, 1, 3, 3, 4)
    message = model.reviser.messages(h, edge, valid)
    assert message[0, 0, 0, 0, 1, 1] == 0
    (message[0, 0, 0, 0, 1, 1] - 1).square().backward()
    assert (edge.grad[0, 0, 1, 1, :, 0, 1] < 0).all()
    restored = edge.detach() - edge.grad
    assert model.reviser.messages(h, restored, valid)[0, 0, 0, 0, 1, 1] > 0


def constant_prediction(model):
    def predict(self, h):
        mean = h.new_tensor([0.0, 4.0]).reshape(1, 1, 2, 1, 1, 1)
        mean = mean.expand(h.shape[0], 1, 2, 3, *h.shape[-2:])
        return mean, torch.zeros_like(mean), h.new_zeros(h.shape[0], 1, 2, 8)

    model.measurement.predict = MethodType(predict, model.measurement)


def sample(frame_id, batch=1, flags=None):
    return MeasurementFrame(
        values=torch.zeros(batch, 3, 8, 8),
        frame_ids=(frame_id,) * batch,
        episode_ids=("withdrawal",) * batch,
        new_measurement=flags,
    )


def test_withdrawal_changes_scores_once_per_new_frame_and_partial_reset_clears_it():
    outputs = {}
    for retention in (0.0, 0.5, 1.0):
        model = N02Encoder(
            model_config(
                update_scale=0, reset_contradicted=False, downweight_contradicted=retention
            )
        )
        constant_prediction(model)
        cold = model.step(sample(0, batch=2), EmbodimentContext(), None, ComputeBudget(4, 4))
        assert cold.representation.revision_energy.count_nonzero() == 0
        out = model.step(sample(1, batch=2), EmbodimentContext(), cold.state, ComputeBudget(4, 4))
        duplicate = model.step(
            sample(1, batch=2), EmbodimentContext(), out.state, ComputeBudget(4, 4)
        )
        torch.testing.assert_close(
            out.representation.weights, duplicate.representation.weights, atol=0, rtol=0
        )
        one_cycle = model.step(
            sample(1, batch=2), EmbodimentContext(), cold.state, ComputeBudget(1, 4)
        )
        torch.testing.assert_close(one_cycle.state.logits, out.state.logits, atol=0, rtol=0)
        mixed = model.step(
            sample(2, batch=2, flags=torch.tensor([True, False])),
            EmbodimentContext(),
            out.state,
            ComputeBudget(4, 4),
        )
        torch.testing.assert_close(mixed.representation.weights[1], out.representation.weights[1])
        assert mixed.representation.revision_energy[1].count_nonzero() == 0
        reset = model.reset(out.state, [True, False])
        assert reset.revision_energy[0].count_nonzero() == 0
        torch.testing.assert_close(reset.revision_energy[1], out.state.revision_energy[1])
        outputs[retention] = out.state.logits
    assert torch.isfinite(outputs[0.0]).all()
    torch.testing.assert_close(
        outputs[0.5][..., 1] - outputs[1.0][..., 1], torch.full((2, 1), -math.log(2))
    )


def test_withdrawal_requires_support_and_a_competing_explanation_within_noise_floor():
    controller = N02Encoder(model_config()).reviser
    score = torch.tensor([0.0, 4.0]).reshape(1, 1, 2, 1, 1)
    assert controller.withdrawal_energy(score, torch.zeros_like(score), temperature=1).sum() == 0
    assert (
        controller.withdrawal_energy(score + 10, torch.ones_like(score), temperature=1).sum() == 0
    )
    penalty = controller.withdrawal_energy(score, torch.ones_like(score), temperature=1)
    assert penalty[0, 0, 1] == pytest.approx(math.log(2))


@pytest.mark.parametrize("K", [1, 4])
def test_actual_score_supervision_remains_finite_when_softmax_underflows(K):
    logits = torch.tensor([[-1000.0, 0.0, -1000.0, 0.0]])[:, :K].requires_grad_()
    rep = SimpleNamespace(
        measurement_energy=torch.zeros(1, 1, K),
        weights=logits.softmax(-1)[:, None],
        posterior_logits=logits[:, None],
        geometry=torch.zeros(1, 1, K, 8),
    )
    out = step_terms(
        rep,
        anchor=torch.tensor([0]),
        wrong=torch.tensor([1]),
        post_contradiction=torch.tensor([True]),
        observations=1,
        diversity_active=False,
    )
    out["mixture"].backward()
    assert torch.isfinite(out["mixture"])
    assert torch.isfinite(logits.grad).all()
    if K > 1:
        assert logits.grad[0, 0] < 0
        assert float(out["mixture"].detach()) == pytest.approx(1000.0)


def test_messages_follow_target_to_source_dependency_and_empty_rows_have_bounded_gradients():
    model = N02Encoder(model_config())
    h = torch.zeros(1, 1, 2, 1, 3, 3)
    h[0, 0, 1, 0, 0, 1] = 8
    edge = torch.zeros(1, 1, 3, 3, 4, 2, 2, requires_grad=True)
    valid = torch.zeros(1, 1, 3, 3, 4)
    valid[0, 0, 1, 1, 0] = 1
    with torch.no_grad():
        edge[0, 0, 1, 1, 0, 0, 1] = 1
    message = model.reviser.messages(h, edge, valid)
    assert message[0, 0, 0, 0, 1, 1] == 8
    assert message[0, 0, 1].count_nonzero() == 0
    message.sum().backward()
    assert edge.grad.abs().max() <= 8


def test_stale_template_helpers_follow_actual_post_change_geometry():
    gen = generator(noise_sigma=0, staleness=1)
    episode = gen.generate(seed=100, prior_condition="stale", identifiable=True)
    assert episode.geometry_changed
    assert episode.value_over_time[0] != episode.value_over_time[-1]
    assert gen.infer_hypothesis(episode, step=7) == episode.value_over_time[-1]
    assert gen.verify_measurement(episode, step=7)["agrees"]


def test_priority_option_controls_accumulation_on_a_returning_path():
    model = N02Encoder(model_config(propagation_hops=2))
    source = torch.zeros(1, 1, 2, 3, 3, dtype=torch.bool)
    source[0, 0, 0, 1, 1] = True
    edges = torch.zeros(1, 1, 3, 3, 4, 2, 2)
    edges[0, 0, 2, 1, 0, 0, 0] = 1
    edges[0, 0, 1, 1, 1, 0, 0] = 1
    valid = torch.ones(1, 1, 3, 3, 4)
    selected, priority = model.reviser.propagate(source, edges, valid)
    model.reviser.revision = replace(model.config.revision, propagate_priority=False)
    max_selected, max_priority = model.reviser.propagate(source, edges, valid)
    assert priority[0, 0, 0, 1, 1] == pytest.approx(1.25)
    assert max_priority[0, 0, 0, 1, 1] == 1
    assert torch.equal(selected, max_selected)

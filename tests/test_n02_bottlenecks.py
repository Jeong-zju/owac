"""Independent numerical and isolation checks for the diagnostic experiment."""

import math

import pytest

torch = pytest.importorskip("torch")

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame  # noqa: E402
from owac.evaluation.n02_bottlenecks import (  # noqa: E402
    calibration,
    paired_episodes,
    physical_frame,
    replay_scores,
    run_episode,
    score_templates,
    tensor_digest,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.model import N02Encoder  # noqa: E402
from owac.training.n02 import episode_prior_context  # noqa: E402


def test_gaussian_log_odds_and_irrelevant_padding():
    # One scalar y=1, means 0/1, sigma=.5: log BF for mean 1 is exactly 2.
    y = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    means = torch.tensor([0.0, 1.0], dtype=torch.float64).reshape(2, 1, 1, 1)
    cost = score_templates(y, means, 0.25, robust="gaussian", reduction="sum")
    assert float(cost[0] - cost[1]) == pytest.approx(2)
    # Add three positions identical under both hypotheses. Sum evidence is
    # invariant; mean evidence is diluted by precisely four.
    padded_y = torch.cat([y, torch.zeros(1, 1, 1, 3)], -1)
    padded_means = torch.cat([means, torch.zeros(2, 1, 1, 3)], -1)
    total = score_templates(padded_y, padded_means, 0.25, robust="gaussian", reduction="sum")
    average = score_templates(padded_y, padded_means, 0.25, robust="gaussian", reduction="mean")
    assert float(total[0] - total[1]) == pytest.approx(2)
    assert float(average[0] - average[1]) == pytest.approx(0.5)


def test_huber_score_is_continuous_at_threshold_and_matches_closed_form():
    y = torch.zeros(1, 1, 1, 1)
    means = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1)
    costs = score_templates(y, means, 1.0, robust="huber", reduction="sum")
    torch.testing.assert_close(costs, torch.tensor([4.5, 7.5]))


def test_replay_respects_prior_and_accumulates_only_recorded_frames():
    record = {
        "prior": [0.475, 0.025, 0.475, 0.025],
        "steps": [
            {"measurement_energy": [2.0, 0.0, 2.0, 0.0], "withdrawal": [0.0] * 4},
            {"measurement_energy": [2.0, 0.0, 2.0, 0.0], "withdrawal": [0.0] * 4},
        ],
    }
    weights = replay_scores(record, 1)
    expected = 1 / (1 + math.exp(math.log(19) - 4))
    assert sum(weights[-1][1::2]) == pytest.approx(expected, abs=1e-6)


def test_no_cue_physical_reference_has_no_semantic_refutation():
    gen, episodes = paired_episodes(190)
    for index in range(8):
        _, templates, mask, _, separation = physical_frame(gen, episodes["unidentifiable"], index)
        assert torch.equal(templates[0], templates[1])
        assert not mask.any()
        assert not separation.any()


def test_physical_reference_matches_production_posterior():
    """Known templates must reproduce the actual encoder's weight update."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = N02Encoder(N02Config(D=8, stem_width=8))
    model.eval()
    gen, episodes = paired_episodes(190)
    episode = episodes["misleading"]
    context = episode_prior_context([episode], 4)
    state = None
    actual = []
    with torch.no_grad():
        reference = next(
            r
            for r in calibration(gen, episode, model)
            if r["mode"] == "grid16_huber_mean_withdrawal"
        )
        for index, values in enumerate(episode.frames):
            _, templates, _, truth, _ = physical_frame(gen, episode, index)
            means = templates[torch.arange(4) % 2][None, None].float()

            def known_prediction(field, means=means):
                return (
                    means,
                    torch.full_like(means, math.log(0.01**2 / 16)),
                    field.new_zeros(1, 1, 4, 8),
                )

            model.measurement.predict = known_prediction
            frame = MeasurementFrame(
                values=torch.from_numpy(values)[None],
                frame_ids=(index,),
                episode_ids=(episode.episode_id,),
                metadata={"variance": 0.01**2},
                metadata_available={"variance": True},
            )
            output = model.step(frame, context, state, ComputeBudget(4, 4))
            state = output.state
            actual.append(float(output.representation.weights[0, 0, truth::2].sum()))
    assert actual == pytest.approx(reference["truth_mass"], abs=2e-6)


def test_rule_interventions_restore_weights_configuration_and_hooks():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = N02Encoder(N02Config(D=8, stem_width=8))
    model.eval()
    gen, episodes = paired_episodes(190)
    # Checkpoints already materialize lazy layers; this fresh test model needs
    # an ordinary forward before freezing its parameter identity.
    with torch.no_grad():
        run_episode(model, gen, episodes["missing"], condition="missing", arm="full")
    before, config = tensor_digest(model), model.config.as_dict()
    original_step = model.reviser.step
    with torch.no_grad():
        base = run_episode(
            model, gen, episodes["missing"], condition="missing", arm="full", forks=True
        )
        for arm in ("no_propagation", "no_withdrawal", "neither", "oracle_filter"):
            run_episode(model, gen, episodes["missing"], condition="missing", arm=arm)
            assert tensor_digest(model) == before
            assert model.config.as_dict() == config
            assert model.reviser.step == original_step
        again = run_episode(model, gen, episodes["missing"], condition="missing", arm="full")
    assert base["steps"] == again["steps"]
    assert len(base["forks"]) == 7

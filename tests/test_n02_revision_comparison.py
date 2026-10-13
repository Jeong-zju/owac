"""Conditional comparison contracts; no hidden truth in semantic scheduling."""

import pytest

torch = pytest.importorskip("torch")

from owac.baselines.n02_measurement_adapters import AdaptedMeasurementScorer  # noqa: E402
from owac.evaluation.n02_revision_comparison import (  # noqa: E402
    absolute_patch_residual,
    cached_selection,
    compose_all,
    evidence_maps,
    make_trial,
    measurement_step,
    update_state,
)
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.revision import RevisionController  # noqa: E402


def controller():
    config = N02Config.from_json("configs/encoders/n02/tiny_contracts.json")
    config.K = 2
    return RevisionController(config).eval()


def test_streaming_matches_causal_prefix_and_duplicate_does_not_recount():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = AdaptedMeasurementScorer(torch.zeros(3), torch.ones(3), "innovation").eval()
    raw = make_trial(26998)["raw"][:2]
    state = None
    for index in range(3):
        state = measurement_step(model, raw[:, index], state, index)
        with torch.no_grad():
            expected = model.scores(raw[:, : index + 1])[:, -1]
        torch.testing.assert_close(state.scores, expected, rtol=1e-5, atol=1e-6)
        assert measurement_step(model, raw[:, index], state, index) is state
    with pytest.raises(ValueError, match="different"):
        measurement_step(model, raw[:, 2] + 1, state, 2)
    with pytest.raises(ValueError, match="ordering"):
        measurement_step(model, raw[:, 1], state, 1)


def test_absolute_fit_rejects_both_bad_even_if_ratio_is_confident():
    trial = make_trial(26998)
    reference = trial["raw"][:, 0]
    normal = absolute_patch_residual(trial["raw"][:, -1], reference)
    assert (normal.min(1).values <= 2.5).all()
    bad = absolute_patch_residual(torch.full_like(reference, 0.6), reference)
    assert (bad > 2.5).all()
    from types import SimpleNamespace

    measurement = SimpleNamespace(scores=torch.tensor([[0.0, 20.0]]).expand(16, 2), residuals=bad)
    assert not controller().semantic_map(evidence_maps(measurement)).any()


def test_one_hop_scheduler_and_cached_dependencies_match_without_changing_inputs():
    from types import SimpleNamespace

    ctrl = controller()
    valid = torch.zeros(4, 4, 4, dtype=torch.bool)
    valid[0, 1, 2] = True  # node1 depends on node0 at its left
    valid[1, 1, 0] = True  # node5 depends on node1's factor, not node1's derived q
    neighbours = [[] for _ in range(16)]
    neighbours[1] = [0]
    neighbours[5] = [1]
    edge = valid[None, None, ..., None, None].float() * torch.eye(2).reshape(1, 1, 1, 1, 1, 2, 2)
    scores = torch.zeros(16, 2)
    scores[0, 1] = 5
    measurement = SimpleNamespace(scores=scores, residuals=torch.full((16, 2), 0.5))
    prior = torch.full((16,), 0.5)
    before = prior.clone()
    derived = compose_all(prior, neighbours)
    outputs = [
        update_state(method, prior, derived, measurement, prior, ctrl, edge, valid, neighbours)
        for method in ("standard_filter", "global_update", "n02_local", "event_cached_filter")
    ]
    torch.testing.assert_close(prior, before, rtol=0, atol=0)
    for factors, state, _ in outputs:
        torch.testing.assert_close(factors, outputs[0][0])
        torch.testing.assert_close(state, outputs[0][1])
    assert outputs[2][2] == outputs[3][2] == 2
    direct = torch.zeros(16, dtype=torch.bool)
    direct[0] = True
    assert cached_selection(direct, neighbours).nonzero().flatten().tolist() == [0, 1]

"""Source provenance, causal evidence, and behavior-gate contracts for NRE controls."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from owac.baselines.n02_ratio import PrefixRatioHead, posterior_from_prefix  # noqa: E402
from owac.baselines.nre_losses import BNRELoss, NRELoss  # noqa: E402
from owac.core.encoder_contract_n02 import (  # noqa: E402
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
)
from owac.evaluation.n02_ratio import (  # noqa: E402
    decision,
    evaluation_data,
    frozen_features,
    protocol,
    summarize_scores,
    training_data,
)
from owac.evaluation.n02_ratio_audit import audit_report  # noqa: E402
from owac.frameworks.n02.config import N02Config  # noqa: E402
from owac.frameworks.n02.model import N02Encoder  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_vendored_class_asts_match_reviewed_upstream_fingerprints():
    tree = ast.parse(Path("src/owac/baselines/nre_losses.py").read_text())
    expected = {
        "NRELoss": "e112b7c9d2f84e7bff1411acfe35f318ef1b69e6fb1421a484a82d62cc832547",
        "BNRELoss": "19f1ddf1edb65e6325f1d8ee878022dd3a0abbe1357cf4ed0435c2c4e62b653c",
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fingerprint = hashlib.sha256(
                ast.dump(node, include_attributes=False).encode()
            ).hexdigest()
            assert fingerprint == expected.pop(node.name)
    assert not expected


def test_loss_keeps_same_class_negative_and_rolls_episode_not_time():
    class Spy(torch.nn.Module):
        def forward(self, theta, features):
            self.received = theta.detach().clone()
            return theta[..., 0, None].expand(-1, -1, features.shape[1]) * features.mean(-1)

    estimator = Spy()
    theta = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    features = torch.zeros(4, 8, 3)
    loss = NRELoss(estimator)(theta, features)
    assert loss.item() == pytest.approx(math.log(2))
    assert estimator.received.shape == (2, 4, 1)
    assert (estimator.received[0] == estimator.received[1]).float().mean() == 0.5


def test_prefix_scores_are_causal_candidate_equivariant_and_target_free():
    torch.manual_seed(23)
    head = PrefixRatioHead(width=4, grid=4, hidden=8).double()
    features = torch.randn(2, 8, 4, 4, 4, dtype=torch.float64)
    scores = head.scores(features)
    altered = features.clone()
    altered[:, 4:] += 100
    torch.testing.assert_close(head.scores(altered)[:, :4], scores[:, :4], rtol=0, atol=0)
    torch.testing.assert_close(head.scores(features[:, :4]), scores[:, :4], rtol=0, atol=1e-14)
    theta = torch.tensor([1.0, -1.0], dtype=torch.float64).reshape(2, 1, 1)
    torch.testing.assert_close(head(theta, features).permute(1, 2, 0), scores.flip(-1))
    # Batch state must reset; an unrelated episode cannot change the first.
    other = features.clone()
    other[1] += 100
    torch.testing.assert_close(head.scores(other)[0], scores[0], rtol=0, atol=0)
    with pytest.raises(ValueError, match="freezes"):
        head.scores(features.requires_grad_())


def test_prefix_replacement_preserves_declared_prior_and_logbf_scale():
    scores = torch.zeros(8, 2, dtype=torch.float64)
    for p in (0.05, 0.5, 0.95):
        torch.testing.assert_close(
            posterior_from_prefix(scores, p), torch.full((8,), p, dtype=torch.float64)
        )
    scores[:, 1] = math.log(19 * 7 / 3)
    expected = torch.full((8,), 0.7, dtype=torch.float64)
    torch.testing.assert_close(posterior_from_prefix(scores, 0.05), expected)
    torch.testing.assert_close(posterior_from_prefix(scores, 0.05), expected)
    # A shared arbitrary offset cannot become absolute fit evidence.
    torch.testing.assert_close(posterior_from_prefix(scores - 1000, 0.05), expected)
    with pytest.raises(ValueError, match="strictly"):
        posterior_from_prefix(scores, 0)


def test_bnre_zero_lambda_equals_nre_and_all_head_branches_receive_gradients():
    torch.manual_seed(19)
    head = PrefixRatioHead(width=4, grid=4, hidden=8)
    features = torch.randn(4, 8, 4, 4, 4)
    theta = torch.tensor([[-1.0], [-1.0], [1.0], [1.0]])
    one, two = NRELoss(head)(theta, features), BNRELoss(head, 0)(theta, features)
    torch.testing.assert_close(one, two, rtol=0, atol=0)
    one.backward()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_balancing_penalty_alone_cannot_certify_no_information_calibration():
    class Bias(torch.nn.Module):
        def forward(self, theta, _features):
            return theta[..., 0] * 3

    theta = torch.tensor([[-1.0], [1.0]] * 4)
    data = torch.zeros(8, 2)
    torch.testing.assert_close(NRELoss(Bias())(theta, data), BNRELoss(Bias())(theta, data))
    p1 = posterior_from_prefix(torch.tensor([[-3.0, 3.0]]))
    assert abs(p1.item() - 0.5) > 0.15


def test_data_independence_and_paired_physical_references():
    data = training_data(range(15980, 15988))
    assert len({row["episode_id"] for row in data["identities"]}) == 8
    assert set(data["truth"].tolist()) == {0, 1}
    evaluation = evaluation_data([17998, 17997])
    for index in (2, 5):
        assert evaluation["zero_prefix"][index].all()
        torch.testing.assert_close(
            evaluation["oracle_scores"][index, :, 0],
            evaluation["oracle_scores"][index, :, 1],
            rtol=0,
            atol=0,
        )
    zero = summarize_scores(torch.zeros_like(evaluation["oracle_scores"]), evaluation)
    assert zero["pair_both_correct"] == 0
    assert zero["no_cue_pass"] == 2
    assert zero["wrong_prior_corrected"] == {"0": 0, "1": 0}
    assert not zero["qualified"]
    audited = audit_report(zero, evaluation["zero_prefix"])
    assert json.loads(json.dumps(audited, allow_nan=False))["pair_both_correct"] == 0


def test_frozen_stem_features_match_actual_production_input_path():
    config = N02Config.from_json(Path("configs/encoders/n02/tiny_contracts.json"))
    model = N02Encoder(config).eval()
    data = training_data([15979])
    cached = frozen_features(model.measurement.stem, data["raw"])
    captured = []
    handle = model.measurement.stem.register_forward_hook(
        lambda _m, _i, output: captured.append(output.detach().clone())
    )
    frame = MeasurementFrame(
        values=data["raw"][:, 0],
        camera_id="synthetic_measurement",
        frame_ids=(0,),
        episode_ids=("comparison",),
        metadata={"variance": 0.0001},
        metadata_available={"variance": True},
    )
    with torch.no_grad():
        model.step(frame, EmbodimentContext(), None, ComputeBudget(4, 4))
    handle.remove()
    torch.testing.assert_close(cached[:, 0], captured[0], rtol=1e-5, atol=1e-6)


def test_gate_requires_same_method_multiple_seeds_both_splits_and_progress():
    declaration = protocol()
    row = {
        "qualified": True,
        "pair_both_correct": 12,
        "no_cue_pass": 12,
        "wrong_prior_corrected": {"0": 4, "1": 4},
    }
    dev = {
        str(i): {m: {"100": copy.deepcopy(row), "200": copy.deepcopy(row)} for m in ("nre", "bnre")}
        for i in range(3)
    }
    held = {str(i): {m: copy.deepcopy(row) for m in ("nre", "bnre")} for i in range(3)}
    assert decision(dev, held, declaration)["nre"]["allow_shared_scorer_comparison"]
    for i in range(2):
        held[str(i)]["nre"]["qualified"] = False
    assert not decision(dev, held, declaration)["nre"]["allow_shared_scorer_comparison"]
    dev["0"]["bnre"]["100"]["pair_both_correct"] = 13
    assert not decision(dev, held, declaration)["bnre"]["allow_shared_scorer_comparison"]

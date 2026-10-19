"""Coexisting causes cannot compete with external-event detection or supervision."""

from types import SimpleNamespace

import pytest
import torch

from owac.evaluation.n04 import evaluate_condition
from owac.frameworks.n04.innovation import InnovationDecomposer
from owac.frameworks.n04.model import N04Encoder
from owac.frameworks.n04.synthetic import SceneConfig
from owac.training.n04 import change_gate_loss, decomposition_loss, load_checkpoint

from .test_six_fixes import config


def inputs():
    return dict(
        current=torch.ones(1, 4, 16, 16),
        predicted=torch.zeros(1, 4, 16, 16),
        validity=torch.ones(1, 1, 16, 16),
        uncertainty=torch.ones(1, 1, 16, 16) * 0.2,
    )


@pytest.mark.parametrize("cause", [0, 2, 3])
def test_other_cause_logit_cannot_reduce_external_event_score(cause):
    dec = InnovationDecomposer(feature_channels=4, hidden=8)
    with torch.no_grad():
        dec.head.bias[1] = 4
        dec.head.bias[cause] = -10
    before = dec(**inputs())
    with torch.no_grad():
        dec.head.bias[cause] = 10
    after = dec(**inputs())
    assert torch.equal(before.gate_change, after.gate_change)
    assert after.gate_change.min() > 0.98
    grad = torch.autograd.grad(after.gate_change.sum(), dec.head.bias)[0]
    assert grad[cause] == 0
    assert grad[1] > 0


def test_evaluation_uses_external_event_score_with_coexisting_imaging_cause():
    encoder = N04Encoder(config())
    with torch.no_grad():
        encoder.decomposer.head.bias[1:3] = 4
    scene = SceneConfig(
        height=16,
        width=16,
        object_velocity=(0.5, 0.5),
        object_colors=((0.4, 0.2, 0.1),),
        noise_sigma=0,
    )
    report = evaluate_condition(
        encoder, condition="object_only", scene=scene, batch=1, length=4, seed=900
    )
    assert report.extra["external_gate_threshold"] == 0.5
    assert report.extra["external_false_negative_rate"] == 0
    out = encoder.decomposer(**inputs())
    assert out.gate_change.min() > 0.98
    assert out.mix[:, 1].max() < 0.5  # this diagnostic share is NOT the detection score


def supervised_fixture():
    valid = torch.ones(1, 1, 16, 16)
    frame = SimpleNamespace(
        measured=valid,
        object_mask=valid,
        true_object_translation=torch.ones(1, 2),
        condition="both",
        metadata={"object_moving": [True]},
        valid_mask=valid,
        saturated_mask=torch.zeros_like(valid),
        oracle_camera_flow=torch.zeros(1, 16, 16, 2),
        true_illumination=torch.tensor([1.5]),
    )
    previous = SimpleNamespace(true_illumination=torch.ones(1))
    logits = torch.tensor([-2.0, 0.0, -2.0, 0.0], requires_grad=True)
    scores = logits.sigmoid()[None, :, None, None].expand(1, 4, 16, 16)
    innovation = SimpleNamespace(
        event_scores=scores,
        mix=scores / scores.sum(1, keepdim=True),
        propagation_validity=valid,
        gate_change=scores[:, 1:2],
        stability=valid * 0.5,
    )
    return frame, previous, logits, innovation


def test_external_event_has_one_target_even_when_other_causes_are_present():
    frame, previous, logits, innovation = supervised_fixture()
    event = change_gate_loss(innovation.gate_change, frame)
    auxiliary = decomposition_loss(innovation, torch.ones(1, 16, 16, 2), frame, previous, config())
    event_grad = torch.autograd.grad(event, logits, retain_graph=True)[0]
    auxiliary_grad = torch.autograd.grad(auxiliary, logits)[0]
    assert event_grad[1] < 0  # moving target asks for a higher external-event score
    assert auxiliary_grad[1] == 0  # no second, normalized target on that score
    assert auxiliary_grad[0] < 0 and auxiliary_grad[2] < 0  # both causes can also be supported


def test_diagnostic_mix_cannot_change_training_loss():
    frame, previous, _, innovation = supervised_fixture()
    flow = torch.ones(1, 16, 16, 2)
    before = decomposition_loss(innovation, flow, frame, previous, config())
    innovation.mix = torch.zeros_like(innovation.mix)
    after = decomposition_loss(innovation, flow, frame, previous, config())
    assert torch.equal(before, after)


def test_disabled_external_branch_does_not_relabel_unknown():
    dec = InnovationDecomposer(feature_channels=4, hidden=8, use_change_branch=False)
    before = dec(**inputs())
    with torch.no_grad():
        dec.head.bias[1] = 20
    after = dec(**inputs())
    assert torch.equal(before.unknown, after.unknown)
    assert after.event_scores[:, 1].sum() == 0
    assert after.gate_change.sum() == 0


def test_old_cause_share_checkpoint_cannot_silently_load_as_event_scores(tmp_path):
    encoder = N04Encoder(config())
    path = tmp_path / "checkpoint.pt"
    torch.save({"encoder": encoder.state_dict(), "mechanism_revision": "n04-six-fixes-v1"}, path)
    with pytest.raises(ValueError, match="incompatible N04 score semantics"):
        load_checkpoint(path, encoder=encoder)

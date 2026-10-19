"""Intervention copies weights and observes encoding without changing it."""

import json
from pathlib import Path

import pytest
import torch

from owac.evaluation.n01_content import ContentProbe, base_config, make_protocol_batch, run_sequence
from owac.evaluation.n01_reliability import copy_intervention, evaluate_case


@pytest.mark.parametrize("mode", ["none", "innovation_risk"])
def test_instrumentation_preserves_task_outputs_and_restores_methods(mode):
    protocol = json.loads(Path("configs/encoders/n01/content_mechanism.json").read_text())
    protocol["evaluation_batch_size"] = 1
    model = ContentProbe(base_config(protocol).replace(time_depths=(8,), source_reliability=mode))
    batch = make_protocol_batch(protocol, 104, "validation", 0, "translation_oracle_alignment")
    model.eval()
    before_step, before_stats = model.encoder.step, model.encoder.integrator.domain_statistics
    with torch.no_grad():
        _, expected = run_sequence(model, batch, protocol)
    actual = evaluate_case(model, batch, protocol)
    assert actual["error_per_member_per_frame"] == expected["error_per_member_per_frame"]
    assert actual["recovery"] == expected["recovery"]
    assert model.encoder.step == before_step
    assert model.encoder.integrator.domain_statistics == before_stats
    assert len(actual["source_diagnostics_per_frame"]) == protocol["length"]
    for record in actual["source_diagnostics_per_frame"]:
        assert 0 <= record["history_mass_mean"] <= 1


def test_intervention_preserves_every_pretrained_tensor():
    protocol = {"width": 8, "size": 16}
    original = ContentProbe(base_config(protocol).replace(time_depths=(8,)))
    checkpoint = {
        "encoder_config": original.encoder.config.to_dict(),
        "model": original.state_dict(),
    }
    risk = copy_intervention(checkpoint, "innovation_risk", 0.01)
    for name, value in original.state_dict().items():
        assert torch.equal(value, risk.state_dict()[name])
    torch.testing.assert_close(risk.encoder.source_reliability_version, torch.tensor([1.0, 0.01]))

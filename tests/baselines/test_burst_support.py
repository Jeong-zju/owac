"""Published parameters stay shared; closed support paths erase dependencies."""

import copy
import json
from pathlib import Path

import torch

from owac.baselines.burst_features import BurstFeatureConfig, BurstFeatureEncoder
from owac.frameworks.n01.support import SupportDomainIntegrator
from owac.frameworks.n01_dbsr import (
    graph_sequence,
    kernel_support,
    support_config,
    supported_backbone,
    supported_fusion,
)


def encoder():
    return BurstFeatureEncoder(
        BurstFeatureConfig(
            stem_width=8, residual_blocks=2, feature_width=16, projection_width=8, offset_width=8
        )
    )


def graph(batch=1, size=12):
    return torch.ones(batch, 4, 2, size, size, dtype=torch.bool)


def test_all_open_full_size_backbone_preserves_upstream_output_and_gradients():
    torch.manual_seed(11)
    first = BurstFeatureEncoder().backbone
    second = copy.deepcopy(first)
    raw = torch.randn(1, 4, 12, 12, requires_grad=True)
    other = raw.detach().clone().requires_grad_(True)
    gates = kernel_support(graph(), SupportDomainIntegrator(support_config()))
    a, b = first(raw), supported_backbone(second, other, gates)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-5)
    a.mean().backward()
    b.mean().backward()
    torch.testing.assert_close(raw.grad, other.grad, atol=1e-7, rtol=2e-5)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=2e-5)


def test_closed_boundary_recomputation_removes_raw_dependency_that_old_features_keep():
    model = encoder()
    for p in model.backbone.parameters():
        p.data.fill_(0.01)
    raw = torch.ones(1, 4, 12, 12, requires_grad=True)
    changed = raw.detach().clone()
    changed[..., 6:] = 2
    opened = graph()
    closed = opened.clone()
    closed[:, :, 0, :, 6] = False
    integrator = SupportDomainIntegrator(support_config())
    before = supported_backbone(model.backbone, raw, kernel_support(opened, integrator))
    before_changed = supported_backbone(model.backbone, changed, kernel_support(opened, integrator))
    recomputed = supported_backbone(model.backbone, raw, kernel_support(closed, integrator))
    recomputed_changed = supported_backbone(
        model.backbone, changed, kernel_support(closed, integrator)
    )
    assert not torch.equal(before[..., :6], before_changed[..., :6])
    assert torch.equal(recomputed[..., :6], recomputed_changed[..., :6])
    recomputed[..., :6].sum().backward()
    assert raw.grad[..., :6].abs().sum() > 0
    assert raw.grad[..., 6:].count_nonzero() == 0


def test_support_also_blocks_hidden_dependencies_in_fusion_weight_predictor():
    torch.manual_seed(19)
    model = encoder()
    features = torch.randn(1, 3, 16, 12, 12)
    other = features.clone()
    other[..., 6:] += 20
    valid = torch.ones(1, 3, 1, 12, 12, dtype=torch.bool)
    offsets = torch.zeros(1, 3, 2, 12, 12)
    closed = graph()
    closed[:, :, 0, :, 6] = False
    gates = kernel_support(closed, SupportDomainIntegrator(support_config()))
    first, weights = supported_fusion(model.fusion, features, valid, offsets, gates)
    second, changed_weights = supported_fusion(model.fusion, other, valid, offsets, gates)
    assert torch.equal(first[..., :6], second[..., :6])
    assert torch.equal(weights[..., :6], changed_weights[..., :6])
    all_open = kernel_support(graph(), SupportDomainIntegrator(support_config()))
    direct, direct_weights = model.fusion(features, valid, offsets)
    equivalent, equivalent_weights = supported_fusion(
        model.fusion, features, valid, offsets, all_open
    )
    torch.testing.assert_close(direct, equivalent, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(direct_weights, equivalent_weights, atol=1e-6, rtol=1e-5)


def test_original_n01_hysteresis_merges_then_refutes_without_future_or_cross_member_data():
    values = torch.full((2, 8, 4, 12, 12), 0.3)
    values[:, 6:, :, :, 6:] = 0.9
    valid = torch.ones_like(values, dtype=torch.bool)
    variance = torch.full_like(values, 0.0001)
    graphs, records = graph_sequence(values, valid, variance, support_config())
    assert graphs[:, 5, :, 0, :, 6].all()
    assert not graphs[:, 6, :, 0, :, 6].any()
    assert sum(records[6]["split_count_per_member"]) > 0
    other = values.clone()
    other[:, 7:] = -1
    other[1] = 1
    alternate, _ = graph_sequence(other, valid, variance, support_config())
    assert torch.equal(graphs[0, :7], alternate[0, :7])
    assert len(set(x for x in records[-1]["unique_source_ids"][0] if x >= 0)) == 8


def test_locked_noise_calibration_controls_flat_pair_refutation_without_a_network():
    protocol = json.loads(
        (Path(__file__).parents[2] / "configs/encoders/n01/burst_support_probe.json").read_text()
    )
    config = support_config(protocol["support_variants"]["noise_calibrated"])
    integrator = SupportDomainIntegrator(config)
    rng = torch.Generator().manual_seed(5501302)
    a, b = torch.randn(2, 65536, generator=rng) * 0.08
    variance = torch.full_like(a, 0.08**2)
    gate = integrator._pair_gate(a, b, variance, variance, torch.ones_like(a, dtype=torch.bool))
    # The one-hop radius-2 distance factor is exp(-1/8). This calibration is
    # pair-level only; it does not promise a graph-level error rate.
    gate = gate * torch.exp(torch.tensor(-1 / 8))
    fraction = float((gate < config.split_threshold).float().mean())
    assert 0.007 < fraction < 0.013
    assert sum(p.numel() for p in integrator.parameters()) == 0

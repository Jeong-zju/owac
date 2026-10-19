"""Upstream equivalence and causal RAW/source boundaries, not quality claims."""

import ast
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from owac.baselines._vendor.dbsr.merging import WeightedSum
from owac.baselines.burst_features import (
    CFA_PHASES,
    BurstFeatureConfig,
    BurstFeatureEncoder,
    MaskedDbsrFusion,
    align_features,
)
from owac.core.encoder_contract import MeasurementFrame

torch.set_num_threads(1)


def config(**kwargs):
    return BurstFeatureConfig(
        stem_width=8,
        residual_blocks=2,
        feature_width=16,
        projection_width=8,
        offset_width=8,
        **kwargs,
    )


def frame(value, frame_id, **kwargs):
    return MeasurementFrame(
        values=value,
        sample_xy=CFA_PHASES,
        frame_ids=frame_id,
        timestamp=frame_id,
        camera_id="c",
        **kwargs,
    )


def test_upstream_executable_definitions_match_pinned_source_digest():
    root = Path(__file__).parents[2] / "src/owac/baselines/_vendor/dbsr"
    manifest = json.loads((root / "provenance.json").read_text())

    class RemoveDocstrings(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return node

    for name, metadata in manifest["files"].items():
        tree = RemoveDocstrings().visit(ast.parse((root / name).read_text()))
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        digest = hashlib.sha256(
            "\n".join(ast.dump(n, include_attributes=False) for n in definitions).encode()
        ).hexdigest()
        assert digest == metadata["executable_ast_sha256"]


@pytest.mark.parametrize("mode", ["dbsr_reference", "dbsr_consensus"])
def test_all_valid_fusion_matches_upstream_outputs_and_gradients(mode):
    torch.manual_seed(91)
    adapted = MaskedDbsrFusion(config(fusion=mode))
    original = WeightedSum(16, 8, 8, offset_modulo=1.0, use_base_frame=mode == "dbsr_reference")
    original.load_state_dict(adapted.state_dict())
    a = torch.randn(2, 4, 16, 8, 8, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    offsets = torch.rand(2, 4, 2, 8, 8) - 0.5
    offsets[:, 0] = 0
    ours, weights = adapted(a, torch.ones(2, 4, 1, 8, 8, dtype=torch.bool), offsets)
    upstream = original(
        {
            "ref_feat": b[:, :1].contiguous(),
            "oth_feat": b[:, 1:].contiguous(),
            "offsets": offsets[:, 1:].contiguous(),
        }
    )
    torch.testing.assert_close(ours, upstream["fused_enc"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(weights, upstream["fusion_weights"], rtol=1e-6, atol=1e-7)
    ours.square().sum().backward()
    upstream["fused_enc"].square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
    for (_, p), (_, q) in zip(adapted.named_parameters(), original.named_parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=3e-6)


def test_reference_backbone_retains_full_capacity_and_exact_upstream_feature_branch():
    model = BurstFeatureEncoder()
    assert model.config.stem_width == 64 and model.config.residual_blocks == 9
    assert model.config.feature_width == 512
    assert sum(p.numel() for p in model.parameters()) > 1_000_000
    raw = torch.randn(1, 2, 4, 12, 12)
    encoded, _ = model.encode_frames(raw, torch.ones_like(raw, dtype=torch.bool))
    up = model.backbone.upstream
    expected = up.out_layer(up.res_layers(up.init_layer(raw.flatten(0, 1))))
    torch.testing.assert_close(encoded.flatten(0, 1), expected)
    assert up.alignment_net is None
    result, _, _ = model(raw, torch.ones_like(raw, dtype=torch.bool))
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("mode", ["current", "mean", "dbsr_reference", "dbsr_consensus"])
def test_invalid_source_values_never_influence_output_and_empty_result_is_zero(mode):
    model = BurstFeatureEncoder(config(fusion=mode))
    x = torch.randn(2, 4, 4, 12, 12)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[:, 2] = False
    mask[1] = False
    corrupt = x.clone()
    corrupt[~mask] = torch.nan
    original = model(x, mask)
    changed = model(corrupt, mask)
    for a, b in zip(original, changed, strict=True):
        torch.testing.assert_close(a, b)
    assert changed[0][1].count_nonzero() == 0
    assert changed[2][1].count_nonzero() == 0
    assert changed[2][:, 2].count_nonzero() == 0
    changed[0].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_missing_reference_uses_available_sources_and_weights_normalize():
    model = MaskedDbsrFusion(config())
    features = torch.randn(1, 3, 16, 8, 8)
    mask = torch.ones(1, 3, 1, 8, 8, dtype=torch.bool)
    mask[:, 0] = False
    out, weights = model(features, mask, torch.zeros(1, 3, 2, 8, 8))
    assert out.abs().sum() > 0 and weights[:, 0].count_nonzero() == 0
    torch.testing.assert_close(weights.sum(1), torch.ones_like(out))
    corrupt = features.clone()
    corrupt[:, 0] = 1e7
    torch.testing.assert_close(model(corrupt, mask, torch.zeros(1, 3, 2, 8, 8))[0], out)


def test_fractional_pull_coordinates_and_positive_tap_validity():
    x = torch.arange(25.0).reshape(1, 1, 1, 5, 5).expand(1, 2, 1, 5, 5).clone()
    mask = torch.ones_like(x, dtype=torch.bool)
    offsets = torch.zeros(1, 2, 2, 5, 5)
    offsets[:, 1, 0] = 0.5
    out, usable, _ = align_features(x, mask, offsets)
    assert out[0, 1, 0, 2, 2] == 12.5
    assert not usable[0, 1, 0, :, -1].any()
    mask[0, 1, 0, 2, 3] = False
    _, usable, _ = align_features(x, mask, offsets)
    assert not usable[0, 1, 0, 2, 2]
    offsets[:, 1, 0] = 0
    offsets[:, 1, 1] = -1
    out, usable, _ = align_features(x, torch.ones_like(mask), offsets)
    assert out[0, 1, 0, 2, 2] == 7
    assert not usable[0, 1, 0, 0].any()


def test_duplicate_partial_reset_and_camera_and_time_boundaries():
    model = BurstFeatureEncoder(config())
    state = model.init_state(2, {"camera_ids": ["c"], "shape": (12, 12)})
    x = torch.randn(2, 4, 12, 12)
    first = model.step(frame(x, 0), state=state)
    repeat = model.step(frame(x + 100, 0), state=first.state)
    assert torch.equal(first.representation, repeat.representation)
    assert torch.equal(repeat.state.filled, torch.ones(2, dtype=torch.long))
    updated = model.step(frame(x + 1, 1), state=repeat.state)
    reset = model.reset_state(updated.state, [True, False])
    assert reset.filled.tolist() == [0, 2]
    fresh = model.step(frame(x, 2), state=reset)
    assert fresh.state.filled.tolist() == [1, 3]
    standalone = model.step(
        frame(x[:1], 2), state=model.init_state(1, {"camera_ids": ["c"], "shape": (12, 12)})
    )
    torch.testing.assert_close(fresh.representation[:1], standalone.representation)
    with pytest.raises(ValueError, match="causal"):
        model.step(frame(x, 0), state=fresh.state)
    with pytest.raises(ValueError, match="camera"):
        model.step(replace(frame(x, 3), camera_id="wrong"), state=fresh.state)


def test_sequence_feature_bank_matches_streaming_and_has_no_future_influence():
    torch.manual_seed(45)
    model = BurstFeatureEncoder(config()).eval()
    values = torch.randn(1, 10, 4, 12, 12)
    mask = torch.ones_like(values, dtype=torch.bool)
    feat, valid = model.encode_frames(values, mask)
    state = model.init_state(1, {"camera_ids": ["c"], "shape": (12, 12)})
    for now in range(10):
        start = max(0, now + 1 - model.config.max_history)
        expected = model.fuse_encoded(
            feat[:, start : now + 1].flip(1), valid[:, start : now + 1].flip(1)
        )[0]
        out = model.step(frame(values[:, now], now), state=state)
        state = model.detach_state(out.state)
        torch.testing.assert_close(out.representation, expected)
    changed = values.clone()
    changed[:, 6:] = 1000
    cf, _ = model.encode_frames(changed, mask)
    torch.testing.assert_close(feat[:, :6], cf[:, :6])
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def test_checkpoints_restore_semantic_configuration_and_reject_missing_phase_contract():
    model = BurstFeatureEncoder(config(fusion="mean"))
    clone = BurstFeatureEncoder(BurstFeatureConfig(**model.config.to_dict()))
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    x = torch.randn(1, 3, 4, 12, 12)
    mask = torch.ones_like(x, dtype=torch.bool)
    torch.testing.assert_close(model(x, mask)[0], clone(x, mask)[0])
    with pytest.raises(ValueError, match="sample_xy"):
        model.step(MeasurementFrame(values=x[:, 0]), state=model.init_state(1, {"shape": (12, 12)}))

"""Independent regressions for the gaps found in the N01 delivery audit."""

import pytest
import torch

from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01.config import N01Config
from owac.frameworks.n01.model import N01Encoder
from owac.frameworks.n01.motion import MotionField
from owac.frameworks.n01.state import init_state, push
from owac.frameworks.n01.support import SupportDomainIntegrator
from owac.training.n01_action import action_masked_loss


def frame(values, number, variance=0.04, fresh=True):
    return MeasurementFrame(
        values,
        frame_ids=torch.full((values.shape[0],), number),
        camera_id="cam",
        timestamp=torch.full((values.shape[0],), float(number)),
        new_measurement=torch.full((values.shape[0],), fresh),
        metadata={"variance": variance},
    )


def test_fractional_overlap_uses_each_original_source_once():
    cfg = N01Config.tiny(
        channels=1,
        spatial_radii=(1,),
        time_depths=(1,),
        neighbourhood_radius=1,
        learnable_gate=False,
        alignment="fixed",
        fractional_alignment=True,
    )
    state = init_state(cfg, 1, {"camera_ids": ["cam"], "shape": (1, 5)})
    state = push(state, frame(torch.tensor([[[[0.0, 1.0, 2.0, 4.0, 8.0]]]]), 0))
    edges = torch.ones(1, 1, 2, 1, 5)
    motion = [
        MotionField(
            torch.zeros(1, 2),
            torch.zeros(1),
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            fractional=torch.tensor([[0.0, 0.5]]),
        )
    ] * 8
    stats, _ = SupportDomainIntegrator(cfg).domain_statistics(state, edges, motion)
    # Three adjacent interpolated targets share two original pixels. The final
    # original-source coefficients are [.5, 1, 1, .5], normalized by their sum 3.
    assert float(stats.mu[0, 0, 0, 0, 2]) == pytest.approx(3.5, rel=1e-5)
    assert float(stats.variance_proxy[0, 0, 0, 0, 2]) == pytest.approx(0.04 * 2.5 / 9, rel=1e-5)
    assert float(stats.unique_samples[0, 0, 0, 0, 2]) == 4
    assert float(stats.unique_frames[0, 0, 0, 0, 2]) == 1
    assert float(stats.weight_concentration[0, 0, 0, 0, 2]) == pytest.approx(9 / 2.5, rel=1e-5)


@pytest.mark.parametrize("planes,channels", [(1, 1), (4, 4)])
def test_random_spatial_temporal_domains_match_all_scalar_statistics(planes, channels):
    cfg = N01Config.tiny(
        input_mode="bayer_cfa" if planes == 4 else "rawvla_rgb3",
        planes=planes,
        channels=channels,
        spatial_radii=(0, 1),
        time_depths=(1, 2, 4),
        neighbourhood_radius=1,
        learnable_gate=False,
    )
    state = init_state(cfg, 2, {"camera_ids": ["cam"], "shape": (3, 4)})
    torch.manual_seed(21)
    for t in range(3):
        f = frame(torch.rand(2, channels, 3, 4), t)
        f.valid_mask = torch.rand_like(f.values) > 0.2
        f.metadata["variance"] = 0.01 + 0.2 * torch.rand_like(f.values)
        state = push(state, f)
    edges = torch.rand(2, planes, 2, 3, 4)
    integral = SupportDomainIntegrator(cfg)
    fast, _ = integral.domain_statistics(state, edges)
    slow = integral.reference_domains(state, edges)
    for name in (
        "mu",
        "variance_proxy",
        "age",
        "support_count",
        "weight_concentration",
        "unique_samples",
        "unique_frames",
    ):
        torch.testing.assert_close(getattr(fast, name), getattr(slow, name), rtol=2e-5, atol=2e-6)


def test_repeated_id_and_padding_preserve_all_eight_unique_frames():
    cfg = N01Config.tiny(
        channels=1, spatial_radii=(0,), time_depths=(1, 4, 8), neighbourhood_radius=0
    )
    state = init_state(cfg, 1, {"camera_ids": ["cam"], "shape": (2, 2)})
    for t in range(8):
        state = push(state, frame(torch.full((1, 1, 2, 2), float(t)), t))
    before = state.values.clone()
    for _ in range(12):
        state = push(state, frame(torch.full((1, 1, 2, 2), 100.0), 7))
    assert torch.equal(state.values, before)
    assert state.frame_ids[:, 0].tolist() == list(range(7, -1, -1))
    stats, _ = SupportDomainIntegrator(cfg).domain_statistics(state, state.support["cam"].edges)
    assert float(stats.unique_frames[-1].min()) == 8


def test_hysteresis_band_remembers_open_and_closed_relations():
    integral = SupportDomainIntegrator(N01Config.tiny(learnable_gate=False))
    strength = torch.full((1, 1, 2, 2, 2), 0.5)
    age = torch.zeros_like(strength)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    measurement = torch.full_like(strength, 0.5)
    integral.update_edges(
        strength, age, measurement, valid, torch.ones_like(strength, dtype=torch.bool)
    )
    assert bool(integral.last_open_edges.all())
    integral.update_edges(
        strength, age, measurement, valid, torch.zeros_like(strength, dtype=torch.bool)
    )
    assert not bool(integral.last_open_edges.any())


def test_zero_budget_and_repeated_frame_do_not_update_support():
    cfg = N01Config.tiny(
        channels=1, spatial_radii=(0, 1), time_depths=(1, 4), neighbourhood_radius=1
    )
    encoder = N01Encoder(cfg).eval()
    state = encoder.init_state(1, {"camera_ids": ["cam"], "shape": (4, 4)})
    f = frame(torch.rand(1, 1, 4, 4), 0)
    output = encoder.step(f, state=state)
    for budget in (None, ComputeBudget(max_updates=0)):
        cached = encoder.step(f, state=output.state, budget=budget)
        assert cached.cost["updates"] == 0
        assert torch.equal(cached.representation, output.representation)
        assert torch.equal(cached.state.support["cam"].edges, output.state.support["cam"].edges)


def test_action_tail_padding_is_excluded_from_flow_loss_and_gradient():
    velocity = torch.zeros(1, 3, 32, requires_grad=True)
    target = torch.ones_like(velocity)
    target[:, 2] = 1000.0
    loss = action_masked_loss(velocity, target, torch.tensor([[True, True, False]]), 7)
    assert float(loss.detach()) == 1.0
    loss.backward()
    assert float(velocity.grad[:, 2].abs().sum()) == 0.0
    assert float(velocity.grad[:, :2].abs().sum()) > 0.0


def test_partial_reset_with_batch_equal_to_history_uses_declared_axes():
    cfg = N01Config.tiny(channels=1, spatial_radii=(0,), time_depths=(1, 8), neighbourhood_radius=0)
    encoder = N01Encoder(cfg).eval()
    state = encoder.init_state(8, {"camera_ids": ["cam"], "shape": (2, 2)})
    output = encoder.step(
        frame(torch.arange(8.0).reshape(8, 1, 1, 1).expand(8, 1, 2, 2), 0), state=state
    )
    state = encoder.reset(
        output.state, torch.tensor([False, True, False, False, False, False, False, False])
    )
    assert not bool(state.observed[:, 1].any())
    assert bool(state.observed[0, 2:].all())
    assert torch.equal(state.values[:, 2:], output.state.values[:, 2:])
    assert torch.equal(state.last_representation[2:], output.state.last_representation[2:])


def test_online_training_draws_new_data_and_aux_target_cannot_enter_buffer():
    from owac.training.n01 import SyntheticSource, build_probe_heads, holdout_mask, sequence_step

    cfg = N01Config.tiny(
        width=8, num_blocks=1, spatial_radii=(0, 1), time_depths=(1, 4), neighbourhood_radius=1
    )
    source = SyntheticSource(cfg, train_seed=2, val_seed=17, image_size=8)
    first, second = source.train_batch(1, 3), source.train_batch(1, 3)
    assert not torch.equal(first.values, second.values)
    assert torch.equal(
        source.val_batch("val_0", 1, 3).values, source.val_batch("val_0", 1, 3).values
    )
    encoder = N01Encoder(cfg).eval()
    heads = build_probe_heads(encoder, (8, 8), channels=3)
    baseline = sequence_step(encoder, heads, first, step=0, grad_enabled=False)
    mask = holdout_mask(8, 8, 0, "cpu", torch.bool)
    first.values[..., mask] = 100.0
    changed = sequence_step(encoder, heads, first, step=0, grad_enabled=False)
    torch.testing.assert_close(baseline.logits, changed.logits, rtol=0, atol=0)
    assert not bool(changed.state.valid[..., mask].any())
    assert float(changed.state.values[..., mask].abs().sum()) == 0.0


def test_cfa_stem_consumes_actual_sampling_offsets_and_quality_masks():
    from owac.frameworks.n01.model import MeasurementStem

    cfg = N01Config.tiny(input_mode="bayer_cfa", channels=4, planes=4, width=8)
    stem = MeasurementStem(cfg)
    values = torch.ones(1, 4, 4, 4)
    valid = torch.ones_like(values, dtype=torch.bool)
    variance = torch.full_like(values, 0.01)
    xy = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    reference = stem(values, valid, variance, xy)
    assert not torch.allclose(reference, stem(values, valid, variance, xy + 1))
    assert not torch.allclose(reference, stem(values, ~valid, variance, xy))


def test_frozen_training_scales_and_partial_reset_have_no_batch_dependence():
    cfg = N01Config.tiny(
        channels=1, width=8, spatial_radii=(0, 1), time_depths=(1, 4), neighbourhood_radius=1
    )
    encoder = N01Encoder(cfg).eval()
    state = encoder.init_state(2, {"camera_ids": ["cam"], "shape": (4, 4)})
    encoder.calibrating = True
    encoder.step(frame(torch.rand(2, 1, 4, 4), 0), state=state)
    encoder.calibrating = False
    saved_scales = encoder.normalization_scales.clone()
    encoder.step(frame(torch.rand(2, 1, 4, 4) * 100, 1), state=state)
    assert torch.equal(encoder.normalization_scales, saved_scales)
    assert "normalization_scales" in encoder.state_dict()


def test_snapshot_manifest_rejects_a_changed_feature_bridge(tmp_path):
    import hashlib
    import json

    from owac.runtime.n01_acceptance import verify

    bridge = tmp_path / "rawvla_features.py"
    bridge.write_text("pinned feature entry")
    (tmp_path / "source-manifest.json").write_text(
        json.dumps({"files": {bridge.name: hashlib.sha256(bridge.read_bytes()).hexdigest()}})
    )
    verify(tmp_path)
    bridge.write_text("unrecorded alteration")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)


def test_author_linear_float_white_ceiling_is_not_invented_black_calibration():
    from owac.raw.rawvla_sequence import RawVLASequenceDataset

    dataset = object.__new__(RawVLASequenceDataset)
    assert (
        dataset._declared_calibration(
            {"representation": "three_channel_linear_float32", "raw_white_level": 3.5}
        )
        is None
    )
    with pytest.raises(ValueError, match="calibration is incomplete"):
        dataset._declared_calibration({"raw_black_level": 0.01})


def test_freezing_backend_preserves_fixed_raw_projection():
    from owac.backends.rawvla_features import RawVLAFeaturePolicy

    policy = object.__new__(RawVLAFeaturePolicy)
    policy.policy = torch.nn.Linear(2, 2)
    policy.encoder = torch.nn.Linear(2, 2).requires_grad_(False)
    policy.postprocessor = torch.nn.Linear(2, 2)
    policy.freeze_backend_parameters()
    assert not any(p.requires_grad for p in policy.encoder.parameters())
    assert not any(p.requires_grad for p in policy.policy.parameters())
    assert all(p.requires_grad for p in policy.postprocessor.parameters())


def test_evidence_exchange_blocks_the_actual_disconnected_neighbor_message():
    from owac.frameworks.n01.model import EvidenceBlock

    cfg = N01Config.tiny(width=8, num_blocks=1)
    block = EvidenceBlock(cfg).eval()
    captured = []
    block.gate[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    features = torch.randn(1, 8, 5, 5)
    changed = features.clone()
    changed[:, :, 2, 1] += 1000.0
    statistics = torch.zeros(1, 6, 5, 5)
    graph = torch.zeros(1, 9, 5, 5)
    graph[:, 4] = 1.0
    block(features, statistics, graph)
    block(changed, statistics, graph)
    torch.testing.assert_close(captured[0][:, :, 2, 2], captured[1][:, :, 2, 2], rtol=0, atol=0)
    graph[:, 3] = 1.0
    block(changed, statistics, graph)
    assert not torch.equal(captured[1][:, :, 2, 2], captured[2][:, :, 2, 2])

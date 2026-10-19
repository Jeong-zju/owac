"""Independent regressions for the N02 acceptance audit (not task-success tests)."""

from dataclasses import replace
from types import MethodType

import pytest
import torch

from owac.backends.rawvla_features_n02 import (
    EncoderReadoutTokenSource,
    RawVLAFeaturePolicy,
    build_stub_bridge,
)
from owac.baselines.n02 import ParticleFilterEncoder, PlainRecurrentEncoder
from owac.core.encoder_contract_n02 import (
    CameraSpec,
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
)
from owac.frameworks.n02.config import MeasurementConfig, N02Config, RevisionConfig, preset
from owac.frameworks.n02.hypotheses import neighbour_stack
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.readout import N02Readout
from owac.training.n02 import step_terms


def config(K=2):
    return N02Config(
        K=K,
        D=8,
        stem_width=8,
        cycles=2,
        beta_prior=0.7,
        temperature=0.5,
        measurement=MeasurementConfig(robust="gaussian"),
        revision=RevisionConfig(update_scale=0.0, contradiction_enabled=False),
    )


def constant_predictor(model):
    def predict(self, h):
        means = torch.arange(h.shape[2], dtype=h.dtype).reshape(1, 1, -1, 1, 1, 1)
        means = means.expand(h.shape[0], 1, h.shape[2], 3, *h.shape[-2:])
        return means, torch.zeros_like(means), torch.zeros(h.shape[0], 1, h.shape[2], 8)

    model.measurement.predict = MethodType(predict, model.measurement)


def frame(index, value=0.6, batch=1, flags=None):
    return MeasurementFrame(
        values=torch.full((batch, 3, 8, 8), value),
        frame_ids=(index,) * batch,
        episode_ids=("audit_episode",) * batch,
        camera_id="libero_agentview",
        new_measurement=flags,
    )


def test_particle_control_uses_same_declared_context_in_initialization_and_proposals():
    model = ParticleFilterEncoder(config())
    state = model.init_state(1, CameraSpec("x", 8, 8, 3, "rawvla_rgb3"), "cpu")
    basis = torch.randn(1, 1, 8, *state.h.shape[-2:])
    a = torch.tensor([[0.95, 0.05, 1.0]])
    b = torch.tensor([[0.05, 0.95, 1.0]])
    initial = model._ensure_particles(state, basis, a)
    proposal = model._propose(basis, 1, 1, a)
    assert not torch.equal(initial, model._ensure_particles(state, basis, b))
    assert not torch.equal(proposal, model._propose(basis, 1, 1, b))
    (initial.square().mean() + proposal.square().mean()).backward()
    for module in (model.initializer, model.proposer):
        assert module.context.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["plain", "particle"])
def test_standalone_controls_initialize_only_the_reset_batch_slot(kind):
    model = PlainRecurrentEncoder(config()) if kind == "plain" else ParticleFilterEncoder(config())
    state = model.init_state(2, CameraSpec("x", 8, 8, 3, "rawvla_rgb3"), "cpu")
    # One episode ended; the other carries a nonzero field.
    state.h[1].fill_(2.0)
    if kind == "plain":
        features = torch.randn(2, model.field_width, *state.h.shape[-2:])
        actual = model._ensure_field(state, features, torch.zeros(2, 2))
    else:
        features = torch.randn(2, 1, model.field_width, *state.h.shape[-2:])
        prior = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        actual = model._ensure_particles(state, features, prior)
    assert actual[0].abs().sum() > 0
    torch.testing.assert_close(actual[1], state.h[1], atol=0, rtol=0)
    assert state.h[0].count_nonzero() == 0


def test_actual_solver_counts_each_likelihood_once_and_does_not_exponentiate_prior():
    model = N02Encoder(config())
    constant_predictor(model)
    budget = ComputeBudget(2, 4)
    prior = torch.tensor([[0.05, 0.95]])
    context = EmbodimentContext(hypothesis_prior=prior)
    state = model.init_state(1, CameraSpec("libero_agentview", 8, 8, 3, "rawvla_rgb3"), "cpu")
    original_h = state.h.clone()
    y = 0.6
    likelihood = -0.5 * (torch.tensor([y, y - 1.0]) ** 2)
    first = model.step(frame(0), context, state, budget)
    torch.testing.assert_close(state.h, original_h, atol=0, rtol=0)
    assert "_buffer" not in state.diagnostics
    second = model.step(frame(1), context, first.state, budget)
    for count, out in ((1, first), (2, second)):
        # Prior is probability mass: E_prior = -T log(p), so tempering the
        # measurement does not silently change the declared prior odds.
        expected = torch.softmax(count * likelihood / 0.5 + 0.7 * prior[0].log(), -1)
        torch.testing.assert_close(out.representation.weights[0, 0], expected)
    before = second.state.h.clone()
    repeated = model.step(frame(1), context, second.state, budget)
    torch.testing.assert_close(repeated.representation.weights, second.representation.weights)
    torch.testing.assert_close(second.state.h, before, atol=0, rtol=0)
    assert first.state.diagnostics["_buffer"].unique_evidence_count == 1
    assert repeated.state.diagnostics["_buffer"].unique_evidence_count == 2
    # Branching the exact same old state must not share a mutable evidence buffer.
    supported = model.step(frame(2, 0.1), context, second.state, budget)
    contradicted = model.step(frame(2, 0.9), context, second.state, budget)
    assert supported.representation.weights[0, 0, 0] > contradicted.representation.weights[0, 0, 0]
    assert second.state.diagnostics["_buffer"].unique_evidence_count == 2


def test_no_new_measurement_for_one_batch_slot_and_partial_reset_keep_other_history():
    model = N02Encoder(config())
    constant_predictor(model)
    budget = ComputeBudget(2, 4)
    old = model.step(frame(0, batch=2), EmbodimentContext(), None, budget)
    mixed = model.step(
        frame(1, batch=2, flags=torch.tensor([True, False])), EmbodimentContext(), old.state, budget
    )
    torch.testing.assert_close(mixed.representation.weights[1], old.representation.weights[1])
    reset = model.reset(mixed.state, torch.tensor([True, False]))
    torch.testing.assert_close(reset.weights[1], mixed.state.weights[1], atol=0, rtol=0)
    assert reset.diagnostics["_buffer"].frames[0].valid_mask[1].all()
    assert not reset.diagnostics["_buffer"].frames[0].valid_mask[0].any()


def test_default_k4_propagation_and_asymmetric_dependency_use_target_to_source_edge():
    model = N02Encoder(config(K=4))
    h = torch.zeros(1, 1, 4, 8, 5, 5)
    state = model.init_state(1, CameraSpec("x", 5, 5, 3, "rawvla_rgb3"), "cpu")
    valid = model._ensure_dependency(state, h)
    residual = torch.zeros(1, 1, 4, 5, 5, dtype=torch.bool)
    residual[0, 0, 1, 2, 2] = True
    selection, _ = model.reviser.propagate(residual, state.dependency_edge, valid)
    assert int(selection.sum()) == 5
    edge = torch.zeros_like(state.dependency_edge)
    # Target (3,2), candidate 0 depends UP on (2,2), candidate 1.
    edge[0, 0, 3, 2, 0, 0, 1] = 1
    selection, _ = model.reviser.propagate(residual, edge, valid)
    assert selection[0, 0, 0, 3, 2]
    assert int(selection.sum()) == 2
    labelled = torch.arange(25).reshape(1, 1, 5, 5)
    assert neighbour_stack(labelled)[0, 0, 3, 2, 0] == labelled[0, 0, 2, 2]


def test_local_revision_preserves_unselected_field_and_input_revision():
    cfg = replace(config(), revision=RevisionConfig())
    model = N02Encoder(cfg)
    state = model.init_state(1, CameraSpec("x", 8, 8, 3, "rawvla_rgb3"), "cpu")
    state.h = torch.randn_like(state.h)
    valid = model._ensure_dependency(state, state.h)
    residual = torch.zeros(1, 1, 2, 8, 8)
    residual[0, 0, 0, 4, 4] = 5
    result = model.reviser.step(
        state,
        residual,
        residual_feature=residual,
        prior_feature=torch.zeros_like(residual),
        location_code=model._position_code(state.h),
        dependency_valid=valid,
    )
    untouched = ~result.propagated.unsqueeze(3).expand_as(state.h)
    torch.testing.assert_close(result.h[untouched], state.h[untouched], atol=0, rtol=0)
    assert state.revision.residual_ema is None
    assert result.reset.any()


def test_geometry_loss_supervises_geometry_head_not_measurement_energy():
    model = N02Encoder(config())
    out = model.step(frame(0), EmbodimentContext(), None, ComputeBudget(2, 4))
    terms = step_terms(
        out.representation,
        anchor=torch.tensor([0]),
        wrong=torch.tensor([1]),
        post_contradiction=torch.tensor([False]),
        observations=1,
        diversity_active=False,
    )
    terms["geometry_set"].backward()
    assert (
        sum(
            float(p.grad.abs().sum())
            for p in model.measurement.head.geometry.parameters()
            if p.grad is not None
        )
        > 0
    )
    assert model.measurement.head.value.weight.grad is None


@pytest.mark.parametrize("name", ["tiny", "small", "medium"])
def test_scale_presets_really_run_one_step(name):
    cfg = preset(name)
    cfg.cycles = 1
    model = N02Encoder(cfg)
    out = model.step(frame(0), EmbodimentContext(), None, ComputeBudget(1, 4))
    assert out.representation.h.shape[2:4] == (cfg.K, cfg.D)


def test_actual_n02_readout_reaches_stub_backend_and_masks_padded_actions():
    cfg = config()
    cfg.token_dim, cfg.tokens_per_camera = 32, 4
    encoder, readout = N02Encoder(cfg), N02Readout(cfg)
    source = EncoderReadoutTokenSource(encoder, readout, token_budget=4)
    base = build_stub_bridge(width=32)["bridge"].base
    bridge = RawVLAFeaturePolicy(
        base, source, token_dim=32, tokens_per_camera=4, image_keys=("libero_agentview",)
    )
    out = bridge.forward(
        observation={"frames": {"libero_agentview": frame(0)}},
        lang_tokens=torch.zeros(1, 2, dtype=torch.long),
        lang_masks=torch.ones(1, 2, dtype=torch.bool),
        state=torch.zeros(1, 32),
        actions=torch.zeros(1, 10, 32),
        action_mask=torch.tensor([[True] * 3 + [False] * 7]),
        return_debug=True,
    )
    manual = (out["u_t"][:, :3] - out["v_t"][:, :3]).square().mean()
    torch.testing.assert_close(out["action_loss"], manual)
    out["action_loss"].backward()
    assert encoder.measurement.stem.input.weight.grad.abs().sum() > 0
    assert readout.tokens.project[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())
    assert source.states["libero_agentview"].step_index == 1


def test_physical_cfa_planes_use_sampling_phase_and_separate_measurement_channels():
    cfg = config()
    cfg.measurement = MeasurementConfig(input_mode="bayer_cfa", cfa_planes=True)
    model = N02Encoder(cfg)
    sample = MeasurementFrame(
        values=torch.rand(1, 4, 8, 8),
        frame_ids=(0,),
        camera_id="cfa",
        sample_xy=torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
    )
    out = model.step(sample, EmbodimentContext(), None, ComputeBudget(2, 4))
    assert model.measurement.predict(out.representation.h)[0].shape[3] == 4
    with pytest.raises(ValueError, match="sample_xy"):
        model.step(
            replace(sample, sample_xy=None), EmbodimentContext(), out.state, ComputeBudget(2, 4)
        )


def test_missing_samples_do_not_enter_grid_average_or_declared_variance():
    model = N02Encoder(config())
    constant_predictor(model)
    values = torch.full((1, 3, 32, 32), float("nan"))
    values[..., 0, 0] = 1.0
    # Real loader uses [B,H,W] masks, with one observed sample in a 4x4 bin.
    valid = torch.zeros(1, 32, 32, dtype=torch.bool)
    valid[:, 0, 0] = True
    measured = replace(
        frame(0),
        values=values,
        valid_mask=valid,
        metadata={"variance": torch.full_like(values, 0.04)},
        metadata_available={"variance": True},
    )
    out = model.step(measured, EmbodimentContext(), None, ComputeBudget(2, 4))
    entry = out.state.diagnostics["_buffer"].frames[0]
    torch.testing.assert_close(entry.values[..., 0, 0], torch.ones(1, 3))
    torch.testing.assert_close(entry.variance[..., 0, 0], torch.full((1, 3), 0.04))
    expected = torch.tensor([0.5 / 0.04, 0.0]) + 0.5 * torch.tensor(0.04).log()
    torch.testing.assert_close(out.representation.measurement_energy[0, 0], expected)
    assert torch.isfinite(out.representation.h).all()
    assert int(out.representation.support.sum()) == 2  # one cell, both candidates


def test_fresh_measurement_features_reach_warm_state_and_partial_reset_is_local():
    cfg = replace(config(), revision=RevisionConfig(contradiction_enabled=False))
    model = N02Encoder(cfg)
    budget = ComputeBudget(2, 4)
    old = model.step(frame(0, batch=2), EmbodimentContext(), None, budget)
    state = model.detach_state(old.state)
    a = model.step(frame(1, 0.1, batch=2), EmbodimentContext(), state, budget)
    b = model.step(frame(1, 0.9, batch=2), EmbodimentContext(), state, budget)
    assert float((a.representation.h - b.representation.h).detach().abs().max()) > 1e-3
    a.representation.h.square().mean().backward()
    assert model.measurement.stem.input.weight.grad.abs().sum() > 0
    reset = model.reset(state, torch.tensor([True, False]))
    next_reset = model.step(frame(1, batch=2), EmbodimentContext(), reset, budget)
    next_normal = model.step(frame(1, batch=2), EmbodimentContext(), state, budget)
    torch.testing.assert_close(
        next_reset.representation.h[1], next_normal.representation.h[1], atol=0, rtol=0
    )


def test_evicted_frame_identity_is_not_accepted_again():
    model = N02Encoder(config())
    constant_predictor(model)
    state = None
    for index in range(8):
        state = model.step(frame(index), EmbodimentContext(), state, ComputeBudget(2, 2)).state
    out = model.step(frame(0), EmbodimentContext(), state, ComputeBudget(2, 2))
    assert not out.representation.diagnostics["new_evidence"]
    assert len(out.state.diagnostics["_buffer"]) == 2
    torch.testing.assert_close(out.representation.weights, state.weights)
    reset = model.reset(out.state, None)
    assert len(reset.diagnostics["_buffer"]) == 0


def test_opaque_frame_ids_are_unique_identifiers_without_lexicographic_time_order():
    model = N02Encoder(config())
    constant_predictor(model)
    old = model.step(frame("uuid-z"), EmbodimentContext(), None, ComputeBudget(2, 1))
    fresh = model.step(frame("uuid-a"), EmbodimentContext(), old.state, ComputeBudget(2, 1))
    assert fresh.representation.diagnostics["new_evidence"]
    duplicate = model.step(frame("uuid-z"), EmbodimentContext(), fresh.state, ComputeBudget(2, 1))
    assert not duplicate.representation.diagnostics["new_evidence"]


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_cfa_loader_descriptor_keeps_canonical_colours_at_actual_phase_origins(pattern):
    from owac.raw.rawvla_sequence_n02 import describe_frame_sampling

    cfg = config()
    cfg.measurement = MeasurementConfig(
        input_mode="bayer_cfa", cfa_planes=True, cfa_pattern=pattern
    )
    model = N02Encoder(cfg)
    sample = replace(
        frame(0),
        values=torch.rand(1, 4, 8, 8),
        sample_xy=describe_frame_sampling(4, cfa_pattern=pattern),
    )
    metadata = model._sampling_metadata(sample, sample.values)
    torch.testing.assert_close(metadata[0, :, :2], torch.as_tensor(sample.sample_xy[:, 0]))
    torch.testing.assert_close(metadata[0, :, 2], torch.tensor([0.0, 1.0, 1.0, 2.0]))
    out = model.step(sample, EmbodimentContext(), None, ComputeBudget(2, 4))
    assert out.representation.h.shape[2] == cfg.K


def test_plain_control_uses_declared_prior_and_matches_actual_trainable_capacity():
    from owac.baselines.n02 import PlainRecurrentEncoder

    cfg = preset("tiny")
    reference = N02Encoder(cfg)
    observation = frame(0)
    budget = ComputeBudget(4, 4)
    reference.step(observation, EmbodimentContext(), None, budget)
    control = PlainRecurrentEncoder(cfg)
    target = sum(p.numel() for p in reference.parameters())
    actual = sum(p.numel() for p in control.parameters())
    assert abs(actual - target) / target < 0.01
    state = control.init_state(1, CameraSpec("libero_agentview", 8, 8, 3, "rawvla_rgb3"), "cpu")
    prior = torch.tensor([[0.475, 0.025, 0.475, 0.025]])
    a = control.step(observation, EmbodimentContext(hypothesis_prior=prior), state, budget)
    b = control.step(observation, EmbodimentContext(hypothesis_prior=prior.flip(-1)), state, budget)
    assert (a.representation.h - b.representation.h).detach().abs().max() > 1e-3
    a.representation.h.square().mean().backward()
    assert control.initializer.context.weight.grad.abs().sum() > 0


def test_revision_loss_retracts_all_redundant_wrong_slots():
    model = N02Encoder(config(K=4))
    out = model.step(frame(0), EmbodimentContext(), None, ComputeBudget(2, 4))
    representation = replace(
        out.representation,
        measurement_energy=torch.tensor([[[10.0, 0.0, 10.0, 0.0]]]),
        weights=torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
    )
    terms = step_terms(
        representation,
        anchor=torch.tensor([0]),
        wrong=torch.tensor([1]),
        post_contradiction=torch.tensor([True]),
        observations=1,
        diversity_active=False,
        diversity_margin=0.1,
    )
    torch.testing.assert_close(terms["revision_wrong_weight"], torch.tensor(0.6))
    torch.testing.assert_close(terms["revision_energy_gap"], torch.tensor(10.0))

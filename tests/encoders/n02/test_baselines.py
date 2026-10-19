"""Contract, energy and ablation tests for the N02 baselines.

Every check here is a real numeric assertion on a real run: no stub encoder, no
fixed-success output, and no tolerance that would pass for a constant tensor.
Three properties are treated as non-negotiable, because the N02 comparison is
invalid without them:

* every arm runs through the shared ``Encoder`` contract and returns a
  ``StructuredRepresentation``;
* every arm's reported measurement energy equals an independent recomputation
  with the shared head and the shared Gaussian NLL, so no arm scored itself with
  a private likelihood;
* a gradient reaches the shared measurement head, so every arm is trainable.

The ablations are checked on the *behaviour* the switch is supposed to change, not
only on the config field: a refutation must actually occur in the scenario used to
show that ``reset_contradicted=False`` stops resetting, and the propagation test
compares against the direct contradiction map.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from owac.baselines.n02 import (
    BASELINE_REGISTRY,
    FixedPriorEncoder,
    PriorCondition,
    ablation_config,
    available_baselines,
    build_baseline,
    declared_prior_energy,
    format_honest_notes,
    honest_notes,
    prior_condition_is_degenerate,
)
from owac.core.encoder_contract_n02 import (
    CameraSpec,
    ComputeBudget,
    CostRecord,
    EmbodimentContext,
    EncoderOutput,
    MeasurementFrame,
    Validity,
)
from owac.frameworks.n02.config import N02Config, RevisionConfig
from owac.frameworks.n02.hypotheses import neighbour_validity
from owac.frameworks.n02.measurement import (
    MeasurementModel,
    compute_measurement_energy,
    gaussian_nll,
)
from owac.frameworks.n02.model import N02Encoder
from owac.frameworks.n02.readout import (
    StructuredRepresentation,
    TokenReadout,
    hypothesis_tokens,
    map_hypothesis,
)
from owac.frameworks.n02.revision import RevisionController
from owac.frameworks.n02.state import HypothesisState

FRAME_SIZE = 8
GRID = (8, 8)
ARMS = available_baselines()
SPEC = CameraSpec(
    camera_id="cam0",
    height=FRAME_SIZE,
    width=FRAME_SIZE,
    channels=3,
    input_mode="rawvla_rgb3",
)


def tiny_config(**overrides) -> N02Config:
    """CPU-sized config: 8x8 frames land on the 8x8 mechanism grid."""
    values: dict = dict(K=4, D=16, stem_width=16, cycles=2, history=2)
    values.update(overrides)
    return N02Config(**values)


def measurement_frame(offset: float, frame_id: int, *, seed: int = 0) -> MeasurementFrame:
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 1, FRAME_SIZE, FRAME_SIZE, generator=generator) * 10.0 + float(offset)
    return MeasurementFrame(
        values=base.repeat(1, 3, 1, 1),
        episode_ids=(0,),
        frame_ids=(frame_id,),
        camera_id="cam0",
        metadata={"frame_id": frame_id},
    )


def run_steps(
    encoder: torch.nn.Module,
    config: N02Config,
    *,
    steps: int = 3,
    offsets: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> tuple[list[EncoderOutput], HypothesisState]:
    state = encoder.init_state(1, SPEC, torch.device("cpu"))
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    outputs: list[EncoderOutput] = []
    for index in range(steps):
        frame = measurement_frame(offsets[index], index, seed=index)
        output = encoder.step(frame, EmbodimentContext(), state, budget)
        state = output.state
        outputs.append(output)
    return outputs, state


@dataclass(slots=True)
class _Prediction:
    """Structural stand-in for ``MeasurementPrediction`` for the energy recheck."""

    mean: torch.Tensor
    variance: torch.Tensor
    geometry: torch.Tensor


def recompute_energy(
    encoder: torch.nn.Module, output: EncoderOutput, config: N02Config
) -> torch.Tensor:
    representation = output.representation
    mean, log_variance, geometry = encoder.measurement.predict(representation.h.detach())
    prediction = _Prediction(mean, torch.exp(log_variance), geometry)
    buffer = output.state.diagnostics["_buffer"]
    buffer = buffer.copy()
    buffer._frames = buffer.frames[-1:]
    result = compute_measurement_energy(prediction, buffer, config.measurement, view_index=0)
    return result.energy


# --------------------------------------------------------------------------------------
# registry and documentation
# --------------------------------------------------------------------------------------


def test_registry_contains_exactly_the_documented_arms():
    required = {
        "single_hypothesis",
        "plain_recurrent",
        "dense_global_recurrent",
        "particle_filter",
        "no_dependency_propagation",
        "no_contradiction_reset",
        "no_proposal",
        "fixed_correct_prior",
        "fixed_misleading_prior",
    }
    assert set(BASELINE_REGISTRY) == required
    assert set(ARMS) == required
    for name, factory in BASELINE_REGISTRY.items():
        assert callable(factory), name


def test_unknown_baseline_is_rejected():
    with pytest.raises(KeyError):
        build_baseline("not_a_baseline", tiny_config())


def test_honest_notes_cover_every_arm_and_render():
    notes = honest_notes()
    assert set(notes) == set(ARMS)
    for name, note in notes.items():
        assert note.name == name
        assert note.shares, name
        assert note.differs, name
        assert note.reference, name
    table = format_honest_notes()
    for name in ARMS:
        assert f"`{name}`" in table
    assert "shares with N02" in table


def test_ablation_factories_change_only_their_declared_switch():
    config = tiny_config()
    assert (
        build_baseline("no_dependency_propagation", config).config.revision.propagation_enabled
        is False
    )
    assert (
        build_baseline("no_contradiction_reset", config).config.revision.reset_contradicted is False
    )
    assert build_baseline("no_proposal", config).config.revision.proposal_enabled is False
    for name in ("no_dependency_propagation", "no_contradiction_reset", "no_proposal"):
        revision = build_baseline(name, config).config.revision
        # The intervention is one flag; nothing else moves with it.
        assert revision.contradiction_enabled is True
        assert revision.write_dependency_edges is True
        assert revision.propagation_hops == config.revision.propagation_hops
    prior = build_baseline("fixed_correct_prior", config)
    assert prior.config.prior_mode == "fixed_odds"
    assert prior.prior_condition is PriorCondition.CORRECT
    assert (
        build_baseline("fixed_misleading_prior", config).prior_condition
        is PriorCondition.MISLEADING
    )
    single = build_baseline("single_hypothesis", config)
    assert single.config.K == 1
    assert single.requested_K == config.K


# --------------------------------------------------------------------------------------
# contract and energy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ARMS)
def test_every_arm_runs_three_steps_through_the_contract(name: str):
    config = tiny_config()
    torch.manual_seed(0)
    encoder = build_baseline(name, config)
    outputs, state = run_steps(encoder, config)
    assert len(outputs) == 3
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    for output in outputs:
        representation = output.representation
        assert isinstance(representation, StructuredRepresentation)
        assert isinstance(output.cost, CostRecord)
        assert isinstance(output.validity, Validity)
        assert isinstance(output.state, HypothesisState)
        weights = representation.weights
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()
        assert weights.shape[-1] == representation.K
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5, rtol=0
        )
        assert torch.isfinite(representation.measurement_energy).all()
        assert torch.isfinite(representation.geometry).all()
        assert tuple(representation.h.shape[-2:]) == GRID
        assert output.cost.updates >= 1
        assert output.cost.updates <= budget.max_updates
        assert output.cost.hypotheses == representation.K
        assert output.cost.history_len >= 1
        assert output.cost.state_bytes is not None and output.cost.state_bytes > 0
    assert state.step_index == 3
    # detach_state must cut the graph at a truncated-BPTT boundary.
    detached = encoder.detach_state(state)
    assert detached.h.requires_grad is False
    # reset clears the selected batch element honestly.
    reset = encoder.reset(state, torch.tensor([True]))
    assert reset.step_index == 0
    assert float(reset.h.abs().sum().item()) == 0.0
    torch.testing.assert_close(
        reset.weights,
        torch.full_like(reset.weights, 1.0 / reset.weights.shape[-1]),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("name", ARMS)
def test_measurement_energy_matches_an_independent_recomputation(name: str):
    """No arm may score itself with a private likelihood.

    The recheck runs immediately after each step: the measurement buffer is a
    state-local object that later steps mutate in place, so an older output's
    energy is only reproducible against the buffer contents of that moment.
    """
    config = tiny_config()
    torch.manual_seed(0)
    encoder = build_baseline(name, config)
    state = encoder.init_state(1, SPEC, torch.device("cpu"))
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    for index in range(3):
        output = encoder.step(
            measurement_frame(float(index), index, seed=index), EmbodimentContext(), state, budget
        )
        state = output.state
        reported = output.representation.measurement_energy[:, 0, :]
        expected = recompute_energy(encoder, output, config)
        torch.testing.assert_close(reported, expected, atol=1e-4, rtol=0)
        assert expected.shape[-1] == output.representation.K


def test_gaussian_nll_convention_is_the_one_the_arms_use():
    """The shared convention: 0.5 * r^2 / v + 0.5 * log(v), Huber capped."""
    residual = torch.tensor([0.0, 1.0, -2.0])
    log_variance = torch.zeros(3)
    expected = 0.5 * residual**2 + 0.5 * log_variance
    torch.testing.assert_close(gaussian_nll(residual, log_variance, robust="gaussian"), expected)
    capped = gaussian_nll(torch.tensor([10.0]), torch.zeros(1), robust="huber", huber_delta=3.0)
    assert float(capped.item()) == pytest.approx(3.0 * (10.0 - 1.5))


@pytest.mark.parametrize("name", ["single_hypothesis", "plain_recurrent"])
def test_k1_arms_report_one_candidate(name: str):
    config = tiny_config(K=4)
    torch.manual_seed(0)
    encoder = build_baseline(name, config)
    outputs, state = run_steps(encoder, config)
    for output in outputs:
        representation = output.representation
        assert representation.K == 1
        assert representation.h.shape[2] == 1
        assert representation.weights.shape[-1] == 1
        assert float(representation.weights.min().item()) == 1.0
        assert output.cost.hypotheses == 1
    assert state.h.shape[2] == 1


@pytest.mark.parametrize("name", ARMS)
def test_map_and_grouped_readouts_work_for_every_arm(name: str):
    config = tiny_config()
    torch.manual_seed(0)
    encoder = build_baseline(name, config)
    outputs, _ = run_steps(encoder, config)
    representation = outputs[-1].representation
    width = int(representation.h.shape[3])
    readout = TokenReadout(width, representation.K, token_dim=32)
    mapped = map_hypothesis(representation, 8, readout=readout, camera_ids=("cam0",))
    grouped = hypothesis_tokens(representation, 8, readout=readout, camera_ids=("cam0",))
    for result in (mapped, grouped):
        assert result.tokens.values.shape[0] == 1
        assert result.tokens.values.shape[-1] == 32
        assert torch.isfinite(result.tokens.values).all()
        assert result.hypothesis_index.max().item() < representation.K
    # The grouped readout must give every candidate its own group; the MAP readout
    # must select exactly one.
    assert grouped.hypothesis_index.unique().numel() == representation.K
    assert mapped.hypothesis_index.unique().numel() == 1
    assert grouped.token_weight_sum.shape[-1] == representation.K


@pytest.mark.parametrize("name", ARMS)
def test_gradients_reach_the_shared_measurement_head(name: str):
    config = tiny_config(cycles=1)
    torch.manual_seed(0)
    encoder = build_baseline(name, config)
    assert isinstance(encoder.measurement, MeasurementModel)
    outputs, _ = run_steps(encoder, config, steps=2, offsets=(0.0, 1.0))
    representation = outputs[-1].representation
    loss = representation.measurement_energy.sum() + representation.geometry.sum()
    loss.backward()
    for parameter_name in ("value.weight", "log_variance.weight"):
        parameter = dict(encoder.measurement.head.named_parameters())[parameter_name]
        assert parameter.grad is not None, f"{name}: {parameter_name} received no gradient"
        assert float(parameter.grad.abs().sum().item()) > 0.0, (
            f"{name}: {parameter_name} gradient is exactly zero"
        )


# --------------------------------------------------------------------------------------
# dense vs local update
# --------------------------------------------------------------------------------------


def test_dense_arm_updates_every_cell_and_the_local_candidate_does_not():
    config = tiny_config(cycles=2)
    torch.manual_seed(0)
    dense = build_baseline("dense_global_recurrent", config)
    dense_outputs, _ = run_steps(dense, config)
    torch.manual_seed(0)
    local = N02Encoder(config)
    local_outputs, _ = run_steps(local, config)
    dense_fractions = [output.cost.updated_fraction for output in dense_outputs]
    local_fractions = [output.cost.updated_fraction for output in local_outputs]
    assert dense_fractions and all(fraction is not None for fraction in dense_fractions)
    assert all(fraction == 1.0 for fraction in dense_fractions), dense_fractions
    assert min(local_fractions) < 1.0, local_fractions
    # Same parameters, same seed, different update allocation: the fields differ.
    assert not torch.allclose(
        dense_outputs[-1].representation.h.detach(), local_outputs[-1].representation.h.detach()
    )


# --------------------------------------------------------------------------------------
# ablations change the behaviour the switch names
# --------------------------------------------------------------------------------------


def _contradicting_config() -> N02Config:
    """Config in which the candidate really refutes cells (measured, not assumed)."""
    return tiny_config(revision=RevisionConfig(relative_margin=1.0, contradiction_threshold=0.5))


def test_contradiction_scenario_actually_refutes_the_candidate():
    """Guard for the two tests below: the scenario must produce resets."""
    config = _contradicting_config()
    torch.manual_seed(0)
    encoder = N02Encoder(config)
    outputs, state = run_steps(encoder, config)
    assert float(state.revision.last_reset.to(torch.float32).mean()) > 0.0
    assert float(state.times_reset.sum().item()) > 0.0
    assert outputs[-1].cost.updated_fraction is not None


def test_no_contradiction_reset_never_sets_a_reset():
    config = _contradicting_config()
    torch.manual_seed(0)
    ablation = build_baseline("no_contradiction_reset", config)
    _, state = run_steps(ablation, config)
    assert state.revision.last_reset is not None
    assert not bool(state.revision.last_reset.any())
    assert float(state.times_reset.sum().item()) == 0.0
    # The refutation itself still happens; only the rebuild is switched off.
    assert float(state.revision.contradiction.to(torch.float32).mean()) > 0.0


def test_no_dependency_propagation_equals_the_direct_contradiction_map():
    config = _contradicting_config()
    torch.manual_seed(0)
    ablation = build_baseline("no_dependency_propagation", config)
    outputs, state = run_steps(ablation, config)
    for output in outputs:
        revision = output.state.revision
        assert torch.equal(revision.propagated, revision.contradiction)
    assert torch.equal(state.revision.propagated, state.revision.contradiction)


def test_propagation_adds_neighbours_beyond_the_direct_contradiction_map():
    """Controller-level check with a single refuted cell, so the union is visible.

    With ``edge = K^2 * I`` each neighbour casts a vote of exactly
    ``propagation_decay``, which is the boundary the controller accepts (``>=``).
    A *block* of refuted cells cannot show the union: every cell around it already
    exceeds the threshold on its own, so the selected set equals the contradiction
    map even though propagation ran. A single cell is therefore used, and the
    expected selected set is that cell plus its four in-grid neighbours.
    """
    config = _contradicting_config()
    on = RevisionController(config, width=config.D)
    off = RevisionController(
        build_baseline("no_dependency_propagation", config).config, width=config.D
    )
    K = config.K
    height = width = 5
    # ``neighbour_validity`` returns ``[4,h,w]``; the neighbour axis must be moved
    # last, because a bare ``view`` scrambles the layout and zeroes interior
    # neighbours that are in fact valid.
    valid = (
        neighbour_validity(height, width, torch.device("cpu"))
        .permute(1, 2, 0)  # [h, w, 4]: the neighbour axis goes LAST
        .unsqueeze(0)
        .unsqueeze(0)
    )
    edge = torch.eye(K).view(1, 1, 1, 1, 1, K, K).repeat(1, 1, height, width, 4, 1, 1) * float(
        K * K
    )
    contradiction = torch.zeros(1, 1, K, height, width, dtype=torch.bool)
    contradiction[0, 0, 0, 2, 2] = True  # one refuted cell, candidate 0
    selection_off, priority_off = off.propagate(contradiction, edge, valid)
    selection_on, priority_on = on.propagate(contradiction, edge, valid)
    # Switching propagation off really does mean "the contradiction map alone".
    assert torch.equal(selection_off, contradiction)
    assert torch.equal(priority_off, contradiction.to(torch.float32))
    # With propagation on, the four in-grid neighbours join the selected set.
    assert int(selection_on.sum()) == 5, int(selection_on.sum())
    assert bool((selection_on & ~contradiction).any())
    neighbours = {(1, 2), (3, 2), (2, 1), (2, 3)}
    selected = {
        (int(y), int(x)) for y, x in (selection_on & ~contradiction)[0, 0, 0].nonzero().tolist()
    }
    assert selected == neighbours, selected
    # The other candidates are untouched by candidate 0's refutation.
    assert not bool(selection_on[0, 0, 1:].any())
    assert float((priority_on - contradiction.to(torch.float32)).max().item()) > 0.0


def test_no_proposal_changes_only_the_cells_selected_for_reset():
    config = _contradicting_config()
    # Identical initialisation for both controllers: the only difference must be
    # the proposal switch, not a different random cell.
    torch.manual_seed(0)
    on = RevisionController(config, width=config.D)
    torch.manual_seed(0)
    off = RevisionController(build_baseline("no_proposal", config).config, width=config.D)
    batch, views, K = 1, 1, config.K
    height = width = 1
    torch.manual_seed(1)
    state = HypothesisState(
        h=torch.randn(batch, views, K, config.D, height, width),
        measurement_energy=torch.zeros(batch, views, K),
        weights=torch.full((batch, views, K), 1.0 / K),
        prior_energy=torch.zeros(batch, views, K),
        relation_energy=torch.zeros(batch, views, K),
        total_energy=torch.zeros(batch, views, K),
        logits=torch.zeros(batch, views, K),
        geometry=torch.zeros(batch, views, K, config.measurement.geometry_dim),
        age=torch.zeros(batch, views, K),
        times_reset=torch.zeros(batch, views, K),
    )
    valid = (
        neighbour_validity(height, width, torch.device("cpu"))
        .permute(1, 2, 0)
        .view(batch, views, height, width, 4)
    )
    state.dependency_valid = valid
    state.dependency_edge = (
        torch.eye(K).view(1, 1, 1, 1, 1, K, K).repeat(batch, views, height, width, 4, 1, 1)
    )
    residual = torch.full((batch, K, height, width), 1.0)
    residual[0, 0] = 5.0
    residual[0, 1] = 3.0
    residual_feature = residual.view(batch, views, K, height, width)
    kwargs = dict(
        residual_feature=residual_feature,
        prior_feature=torch.zeros(batch, views, K, height, width),
        location_code=torch.zeros(batch, views, 4, height, width),
        dependency_valid=valid,
        iteration=0,
        cold_start=False,
    )
    # ``HypothesisUpdateCell`` holds LazyConv2d layers, so its weights are
    # materialised on the first forward: seed immediately before each call so both
    # controllers really do share one initialisation.
    torch.manual_seed(0)
    out_on = on.step(state, residual.unsqueeze(1), **kwargs)
    state_off = HypothesisState(
        h=state.h.clone(),
        measurement_energy=state.measurement_energy.clone(),
        weights=state.weights.clone(),
        prior_energy=state.prior_energy.clone(),
        relation_energy=state.relation_energy.clone(),
        total_energy=state.total_energy.clone(),
        logits=state.logits.clone(),
        geometry=state.geometry.clone(),
        age=state.age.clone(),
        times_reset=state.times_reset.clone(),
        dependency_edge=state.dependency_edge.clone(),
        dependency_valid=valid.clone(),
    )
    torch.manual_seed(0)
    out_off = off.step(state_off, residual.unsqueeze(1), **kwargs)
    # ``HypothesisUpdateCell`` holds LazyConv2d layers, so their weights only exist
    # after a first forward pass. Both controllers must see the same RNG state for
    # that pass; otherwise their cell weights differ and the comparison would be
    # measuring two different networks instead of the proposal switch.
    zeros = torch.zeros(batch, views, K, config.D, height, width)
    evidence = torch.zeros(batch, views, K, height, width)
    position = torch.zeros(batch, views, 4, height, width)
    gate_input = torch.zeros(batch * views, 3, height, width)
    torch.manual_seed(0)
    on.cell(zeros, zeros, evidence, evidence, position)
    on.location_update(gate_input)
    # Materialise the ablation's lazy layers from the *same* resolved weights: the
    # two controllers are the same network with one switch flipped, and the test
    # must not accidentally compare two different random initialisations.
    off.cell.load_state_dict(on.cell.state_dict())
    off.location_update.load_state_dict(on.location_update.state_dict())
    assert torch.equal(on.location_update.weight, off.location_update.weight)
    assert all(
        torch.equal(left, right)
        for left, right in zip(on.cell.parameters(), off.cell.parameters(), strict=True)
    )
    torch.manual_seed(0)
    out_on = on.step(state, residual.unsqueeze(1), **kwargs)
    torch.manual_seed(0)
    out_off = off.step(state_off, residual.unsqueeze(1), **kwargs)
    reset = out_on.reset
    assert bool(reset.any()), "the scenario must select at least one cell for reset"
    outside = ~reset.unsqueeze(3).expand_as(out_on.h)
    torch.testing.assert_close(out_off.h[outside], out_on.h[outside], atol=0, rtol=0)
    assert not torch.allclose(
        out_off.h[reset.unsqueeze(3).expand_as(out_on.h)],
        out_on.h[reset.unsqueeze(3).expand_as(out_on.h)],
    )
    # The proposal parameters are still built, so the parameterisation matches N02.
    assert any(parameter.numel() > 0 for parameter in off.proposal.parameters())


def test_no_proposal_changes_the_learned_trajectory():
    config = _contradicting_config()
    torch.manual_seed(0)
    baseline = N02Encoder(config)
    baseline_outputs, baseline_state = run_steps(baseline, config)
    torch.manual_seed(0)
    ablation = build_baseline("no_proposal", config)
    ablation_outputs, ablation_state = run_steps(ablation, config)
    assert float(baseline_state.times_reset.sum().item()) > 0.0
    # The switch disables the rebuild, not the detection: refuted cells are still
    # marked and still selected for reset.
    assert float(ablation_state.times_reset.sum().item()) > 0.0
    assert not torch.allclose(
        baseline_outputs[-1].representation.h.detach(),
        ablation_outputs[-1].representation.h.detach(),
    )


# --------------------------------------------------------------------------------------
# declared priors
# --------------------------------------------------------------------------------------


def test_declared_prior_energy_encodes_the_condition():
    correct = declared_prior_energy(PriorCondition.CORRECT, 1.0, 4, favoured_index=0)
    torch.testing.assert_close(correct, torch.tensor([-1.0, 0.0, 0.0, 0.0]))
    misleading = declared_prior_energy(PriorCondition.MISLEADING, 1.0, 4, favoured_index=0)
    torch.testing.assert_close(misleading, torch.tensor([0.0, -1.0, 0.0, 0.0]))
    for condition in (PriorCondition.UNINFORMATIVE, PriorCondition.MISSING):
        torch.testing.assert_close(declared_prior_energy(condition, 1.0, 4), torch.zeros(4))
    torch.testing.assert_close(
        declared_prior_energy(PriorCondition.STALE, 0.5, 2, favoured_index=1),
        torch.tensor([0.0, -0.5]),
    )
    with pytest.raises(ValueError):
        declared_prior_energy(PriorCondition.CORRECT, -1.0, 4)
    with pytest.raises(ValueError):
        declared_prior_energy("not_a_condition", 1.0, 4)


def test_fixed_priors_are_declared_input_and_never_measurement():
    config = tiny_config()
    torch.manual_seed(0)
    correct = build_baseline("fixed_correct_prior", config, level=1.0, favoured_index=0)
    torch.manual_seed(0)
    misleading = build_baseline("fixed_misleading_prior", config, level=1.0, favoured_index=0)
    correct_state = correct.init_state(1, SPEC, torch.device("cpu"))
    misleading_state = misleading.init_state(1, SPEC, torch.device("cpu"))
    # The declaration is visible in the state's prior term before any measurement.
    correct_prior = correct._prior_energy(correct_state, cold_start=True)
    misleading_prior = misleading._prior_energy(misleading_state, cold_start=True)
    torch.testing.assert_close(correct_prior[0, 0], torch.tensor([-1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(misleading_prior[0, 0], torch.tensor([0.0, -1.0, 0.0, 0.0]))
    scale = correct._bounded_beta_prior() / config.temperature
    torch.testing.assert_close(
        correct.effective_log_odds, torch.tensor([1.0, 0.0, 0.0, 0.0]) * scale
    )
    correct_outputs, correct_final = run_steps(correct, config)
    misleading_outputs, misleading_final = run_steps(misleading, config)
    # Same frames, same seed: the measurement evidence is identical, so the buffer
    # cannot contain the declaration.
    correct_buffer = correct_final.diagnostics["_buffer"]
    misleading_buffer = misleading_final.diagnostics["_buffer"]
    assert [frame.identity for frame in correct_buffer.frames] == [
        frame.identity for frame in misleading_buffer.frames
    ]
    torch.testing.assert_close(
        correct_buffer.stack_values(), misleading_buffer.stack_values(), atol=0, rtol=0
    )
    assert correct_outputs[-1].representation.diagnostics["prior_condition"] == "correct"
    assert misleading_outputs[-1].representation.diagnostics["prior_condition"] == "misleading"
    # The declaration is an intervention that reaches the posterior.
    assert not torch.allclose(
        correct_outputs[-1].representation.weights.detach(),
        misleading_outputs[-1].representation.weights.detach(),
    )


def test_fixed_prior_direction_is_analytic_when_the_field_cannot_move():
    """With the update blend forced to zero the score difference is exact.

    The declared prior must move the score by ``-beta_prior * offset / temperature``
    and by nothing else; any leak into the measurement term would break this.
    """
    config = ablation_config(tiny_config(cycles=1), contradiction_enabled=False, update_scale=0.0)
    torch.manual_seed(0)
    correct = build_baseline("fixed_correct_prior", config, level=1.0, favoured_index=0)
    torch.manual_seed(0)
    misleading = build_baseline("fixed_misleading_prior", config, level=1.0, favoured_index=0)
    for encoder in (correct, misleading):
        with torch.no_grad():
            encoder.reviser.location_update.bias.fill_(-20.0)
            encoder.reviser.location_update.weight.zero_()
    correct_outputs, correct_final = run_steps(correct, config, steps=1, offsets=(0.0,))
    misleading_outputs, misleading_final = run_steps(misleading, config, steps=1, offsets=(0.0,))
    correct_logits = correct_outputs[-1].state.logits[0, 0]
    misleading_logits = misleading_outputs[-1].state.logits[0, 0]
    torch.testing.assert_close(
        correct_outputs[-1].representation.h.detach(),
        misleading_outputs[-1].representation.h.detach(),
        atol=1e-5,
        rtol=0,
    )
    # Identical fields and identical evidence, so the whole difference is declared.
    torch.testing.assert_close(
        correct_final.prior_energy[0, 0] - misleading_final.prior_energy[0, 0],
        torch.tensor([-1.0, 1.0, 0.0, 0.0]),
    )
    scale = correct._bounded_beta_prior() / config.temperature
    torch.testing.assert_close(
        correct_logits - misleading_logits,
        torch.tensor([1.0, -1.0, 0.0, 0.0]) * scale,
        atol=1e-4,
        rtol=0,
    )
    assert float(correct_outputs[-1].representation.weights[0, 0, 0]) > float(
        misleading_outputs[-1].representation.weights[0, 0, 0]
    )
    assert float(misleading_outputs[-1].representation.weights[0, 0, 1]) > float(
        correct_outputs[-1].representation.weights[0, 0, 1]
    )


def test_misleading_prior_with_one_candidate_is_reported_as_degenerate():
    config = tiny_config(K=1)
    encoder = FixedPriorEncoder(config, PriorCondition.MISLEADING, level=1.0)
    assert encoder.prior_degenerate is True
    assert prior_condition_is_degenerate(PriorCondition.MISLEADING, 1) is True
    assert prior_condition_is_degenerate(PriorCondition.MISLEADING, 4) is False
    assert prior_condition_is_degenerate(PriorCondition.CORRECT, 1) is False
    output, state = run_steps(encoder, config, steps=1, offsets=(0.0,))
    assert output[0].representation.diagnostics["prior_degenerate"] is True
    assert encoder.prior_condition.available is True
    missing = FixedPriorEncoder(tiny_config(), PriorCondition.MISSING, level=1.0)
    assert missing.prior_condition.available is False
    missing_output, _ = run_steps(missing, tiny_config(), steps=1, offsets=(0.0,))
    assert missing_output[0].representation.diagnostics["prior_available"] is False
    assert state.h.shape[2] == 1


# --------------------------------------------------------------------------------------
# particle filter
# --------------------------------------------------------------------------------------


def test_particle_filter_weights_follow_the_measurement():
    config = tiny_config(K=4, cycles=2, temperature=4.0)
    torch.manual_seed(0)
    encoder = build_baseline("particle_filter", config, resample_jitter=False)
    outputs_a, _ = run_steps(encoder, config)
    outputs_b, _ = run_steps(encoder, config, offsets=(5.0, 7.5, 9.0))
    weights_a = outputs_a[-1].representation.weights
    weights_b = outputs_b[-1].representation.weights
    assert torch.isfinite(weights_a).all() and torch.isfinite(weights_b).all()
    torch.testing.assert_close(
        weights_a.sum(dim=-1), torch.ones_like(weights_a.sum(dim=-1)), atol=1e-5, rtol=0
    )
    assert float((weights_a - weights_b).abs().max().item()) > 1e-3
    assert (
        float(
            (
                outputs_a[-1].representation.measurement_energy
                - outputs_b[-1].representation.measurement_energy
            )
            .abs()
            .max()
            .item()
        )
        > 1e-3
    )


def test_particle_filter_resampling_keeps_the_candidate_count():
    config = tiny_config(K=4, cycles=2)
    torch.manual_seed(0)
    encoder = build_baseline("particle_filter", config, propose_ratio=0.5, resample_jitter=False)
    outputs, state = run_steps(encoder, config)
    for output in outputs:
        assert output.representation.K == 4
        assert output.representation.h.shape[2] == 4
        assert output.representation.weights.shape[-1] == 4
        proposed = output.diagnostics["proposed_last"]
        resampled = output.diagnostics["resampled_last"]
        assert proposed + resampled == 4
        assert output.diagnostics["proposal_counts"] == [2, 1]
    # 2 proposed slots in cycle 0 and 1 in cycle 1, for each of the three steps.
    assert float(state.times_reset.sum().item()) == 3.0 * 3.0
    # The extremes must also preserve the count.
    for propose_ratio, expected in ((0.0, 0), (1.0, 4)):
        torch.manual_seed(0)
        edge = build_baseline(
            "particle_filter", config, propose_ratio=propose_ratio, resample_jitter=False
        )
        edge_outputs, _ = run_steps(edge, config, steps=1, offsets=(0.0,))
        assert edge_outputs[0].representation.K == 4
        assert edge_outputs[0].diagnostics["proposed_last"] == expected
        assert edge_outputs[0].diagnostics["resampled_last"] == 4 - expected


def test_particle_filter_stop_gradient_switch_cuts_the_resampled_branch():
    config = tiny_config(K=4, cycles=1)
    torch.manual_seed(0)
    cut = build_baseline("particle_filter", config, stop_resample_gradient=True)
    cut_outputs, _ = run_steps(cut, config, steps=1, offsets=(0.0,))
    assert cut_outputs[0].representation.h.requires_grad is True  # the head path stays live
    torch.manual_seed(0)
    live = build_baseline("particle_filter", config, stop_resample_gradient=False)
    live_outputs, _ = run_steps(live, config, steps=1, offsets=(0.0,))
    assert live_outputs[0].diagnostics["stop_resample_gradient"] is False
    assert cut_outputs[0].diagnostics["stop_resample_gradient"] is True
    # Both arms still train through the shared head.
    for output, encoder in ((cut_outputs[0], cut), (live_outputs[0], live)):
        encoder.zero_grad(set_to_none=True)
        output.representation.measurement_energy.sum().backward()
        grad = dict(encoder.measurement.head.named_parameters())["value.weight"].grad
        assert grad is not None and float(grad.abs().sum().item()) > 0.0


def test_particle_filter_rejects_invalid_options():
    config = tiny_config()
    with pytest.raises(ValueError):
        build_baseline("particle_filter", config, propose_ratio=1.5)
    encoder = build_baseline("particle_filter", config)
    weights = torch.full((1, 1, 4), 0.25)
    with pytest.raises(ValueError):
        encoder._systematic_resample(weights, 5)


def test_particle_filter_systematic_resample_matches_the_cumulative_weights():
    encoder = build_baseline("particle_filter", tiny_config(K=4), resample_jitter=False)
    weights = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])
    indices = encoder._systematic_resample(weights, 4)
    assert indices.shape == (1, 1, 4)
    assert indices.flatten().tolist() == [2, 2, 2, 2]
    uniform = torch.full((1, 1, 4), 0.25)
    spread = encoder._systematic_resample(uniform, 4)
    assert sorted(spread.flatten().tolist()) == [0, 1, 2, 3]


def test_plain_recurrent_has_no_hypothesis_or_revision_state():
    config = tiny_config()
    torch.manual_seed(0)
    encoder = build_baseline("plain_recurrent", config)
    outputs, state = run_steps(encoder, config)
    representation = outputs[-1].representation
    assert representation.K == 1
    assert torch.equal(representation.weights, torch.ones_like(representation.weights))
    assert torch.equal(representation.prior_energy, torch.zeros_like(representation.prior_energy))
    assert representation.h.shape[1] == 1
    assert state.dependency_edge is None
    assert state.dependency_valid is None
    assert float(state.times_reset.sum().item()) == 0.0
    # The ConvGRU really does move the field (an identity update would be a stub).
    assert (
        float((outputs[-1].representation.h - outputs[0].representation.h).abs().max().item()) > 0.0
    )
    assert float((outputs[-1].representation.h - state.h).abs().max().item()) == 0.0

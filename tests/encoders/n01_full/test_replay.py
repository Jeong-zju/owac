"""Replay semantics: full solve, fixed-decision replay, counterfactual replay.

The counterfactual comparison is made against the independent NumPy FP64 oracle,
and the fixed-decision interface is checked with a deliberately leaky control so
"unchanged" cannot pass by construction.  The dependency-closure property is
exercised on the two-layer nonlinear toy operator, where the documented wrong
version (zero the removed input) provably differs.
"""

import pytest
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full import reference as reference_module
from owac.frameworks.n01_full import replay as replay_module
from owac.frameworks.n01_full.config import ReplayConfig
from owac.frameworks.n01_full.contracts import DependencyNode, SourceKey
from owac.frameworks.n01_full.domains import make_domain, retract_sources
from owac.frameworks.n01_full.ledger import ContentReplacement, MeasurementLedger

DATASET = "replay"


def observe_event(values, frame_id, *, episode_id="e0", camera_id="c0"):
    tensor = torch.as_tensor(values, dtype=torch.float64).reshape(1, 1, 1, -1)
    return replay_module.ObserveEvent(
        MeasurementFrame(
            values=tensor,
            episode_ids=[episode_id],
            frame_ids=[frame_id],
            camera_id=camera_id,
            timestamp=torch.tensor([float(frame_id)]),
            new_measurement=torch.tensor([True]),
        )
    )


def stream(frames):
    return replay_module.EventStream(
        DATASET,
        tuple(
            observe_event([float(index + 1), float(index + 2)], index) for index in range(frames)
        ),
    )


def all_keys(frames):
    return [
        SourceKey(DATASET, "e0", "c0", frame, 0, 0, x) for frame in range(frames) for x in (0, 1)
    ]


def builder_for(keys):
    def builder(ledger: MeasurementLedger):
        evidence = [key for key in keys if ledger.has_source(key)]
        return [
            make_domain(
                "D",
                reference_chart="ref",
                basis="phase_local_constant",
                phase=0,
                evidence_source_ids=evidence,
                decision_version="dv1",
                mapping_version="mv1",
            )
        ]

    return builder


def grid():
    return replay_module.ReadoutGrid((replay_module.QueryPoint("q0", "D", 0, 0.0, 0.0),))


def oracle_reference(ledger, domains, readout_grid):
    out = {}
    for domain in domains:
        observations = [
            reference_module.ReferenceObservation(
                source_id=key,
                phase=key.channel_or_phase,
                xy=ledger.sample_coordinates(key),
                value=float(ledger.get_source(key).raw_value),
                variance=1.0,
            )
            for key in domain.evidence_source_ids
            if ledger.has_source(key) and ledger.get_source(key).validity_kind == "valid"
        ]
        fit = reference_module.reference_fit(observations, basis="constant", dimension=2)
        for point in readout_grid.points:
            if point.domain_id == domain.domain_id:
                out[point.label] = fit.predict(point.phase, (point.x, point.y))[0]
    return out


def test_full_replay_commits_versions_and_exact_range():
    result = replay_module.full_replay(
        stream(3), domain_builder=builder_for(all_keys(3)), readout_grid=grid()
    )
    versions = result.snapshot.versions
    assert set(versions) == {
        "parameter_version",
        "support_version",
        "mapping_version",
        "calibration_version",
    }
    assert result.exact_replay_range.unique_frames == 3
    assert result.exact_replay_range.covers(3)
    assert not result.exact_replay_range.covers(4)
    assert result.history_loss == ()
    assert result.fits[0].estimate.independent_observation_count == 6
    # The commit is one object: readout, versions, range and cost belong together.
    assert float(result.snapshot.representation[0]) == result.readout.means[0]


def test_fixed_decision_replay_is_falsifiable():
    events = stream(3)
    keys = all_keys(3)
    full = replay_module.full_replay(events, domain_builder=builder_for(keys), readout_grid=grid())
    target = keys[3]
    replacement = ContentReplacement("garbage", {target: 1234.0})
    clean = replay_module.fixed_decision_replay(
        events,
        frozen_domains=builder_for(keys)(full.ledger.without([target])),
        retracted_ids=[target],
        replacement=replacement,
        readout_grid=grid(),
    )
    assert clean.unchanged
    assert clean.max_abs_diff == 0.0
    assert "counterfactual_replay" in " ".join(clean.manifest["not_claimed"])

    # Negative control: decisions that still claim the retracted source as evidence
    # must show the leak; otherwise the interface would be self-confirming.
    leaky = replay_module.fixed_decision_replay(
        events,
        frozen_domains=builder_for(keys)(full.ledger),
        retracted_ids=[target],
        replacement=replacement,
        readout_grid=grid(),
    )
    assert not leaky.unchanged
    assert leaky.max_abs_diff > 1.0
    # The raw record and the frozen decisions are untouched by the replay.
    assert full.ledger.get_source(target).raw_value == 3.0
    assert leaky.frozen_domain_versions == (builder_for(keys)(full.ledger)[0].support_version,)


def test_counterfactual_replay_matches_independent_batch_reference():
    events = stream(3)
    keys = all_keys(3)
    target = keys[3]
    result = replay_module.counterfactual_replay(
        events,
        domain_builder=builder_for(keys),
        retracted_ids=[target],
        readout_grid=grid(),
        reference_fn=oracle_reference,
    )
    assert result.matches_reference is True
    assert result.claim == "exact_independence_in_event_stream"
    assert not result.beyond_exact_range
    assert result.result.fits[0].estimate.independent_observation_count == 5
    assert target not in result.result.domains[0].evidence_source_ids


def test_counterfactual_replay_removes_only_the_named_source():
    events = stream(3)
    keys = all_keys(3)
    target = keys[0]
    result = replay_module.counterfactual_replay(
        events, domain_builder=builder_for(keys), retracted_ids=[target], readout_grid=grid()
    )
    members = set(result.result.domains[0].evidence_source_ids)
    assert target not in members
    assert set(keys) - {target} == members
    # The exposure still exists; only the sample is absent from every query.
    assert result.result.ledger.unique_frame_count(camera_id="c0") == 3
    assert not result.result.ledger.has_source(target)


def test_exact_window_boundary_records_history_loss_and_rebuild_claim():
    events = stream(10)
    keys = [SourceKey(DATASET, "e0", "c0", frame, 0, 0, 0) for frame in range(10)]
    config = ReplayConfig(exact_window_unique_frames=8)
    full = replay_module.full_replay(
        events, domain_builder=builder_for(keys), readout_grid=grid(), replay_config=config
    )
    assert len(full.exact_window_frames) == 8
    assert len(full.history_loss) == 2
    assert {loss.reason for loss in full.history_loss} == {"beyond_exact_window_summary_only"}
    beyond = replay_module.counterfactual_replay(
        events,
        domain_builder=builder_for(keys),
        retracted_ids=[keys[0]],
        readout_grid=grid(),
        replay_config=config,
    )
    assert beyond.beyond_exact_range
    assert beyond.claim == "rebuild_from_legal_boundary"
    assert beyond.manifest["invalidated_summary"] is True
    assert beyond.manifest["not_claimed"], (
        "an out-of-window source must not claim exact equivalence"
    )
    inside = replay_module.counterfactual_replay(
        events,
        domain_builder=builder_for(keys),
        retracted_ids=[keys[-1]],
        readout_grid=grid(),
        replay_config=config,
    )
    assert not inside.beyond_exact_range


def test_dependency_closure_recomputation_matches_full_retraction():
    program = replay_module.ToyProgram(
        ["s0", "s1", "s2"],
        hidden=torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.75]], dtype=torch.float64),
        output=torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float64),
    )
    values = {"s0": 1.0, "s1": 5.0, "s2": -2.0}
    full = program.evaluate(values)
    truth = program.retract(values, ["s1"])
    closure, recomputed = replay_module.recompute_successor_closure(program, full, ["s1"])
    assert float((closure["readout"] - truth["readout"]).abs().max()) <= 1e-12
    for node in ("normalization", "hidden", "output", "readout", "centered.0", "centered.2"):
        assert node in recomputed
    assert "centered.1" not in closure  # the removed source's centered value is gone
    # The wrong version: keep the graph and zero the removed input. It changes the
    # normalization, so it must not be accepted as an equivalent recomputation.
    zeroed = program.zero_input(values, ["s1"])
    assert float((zeroed["readout"] - truth["readout"]).abs().max()) > 1e-6
    # The dependency graph records all four versions and full parents.
    nodes = program.dependency_nodes()
    assert nodes["hidden"].parent_ids == ("centered.0", "centered.1", "centered.2")
    assert nodes["readout"].is_stale_against({"parameter_version": "other"})
    assert not nodes["readout"].is_stale_against(
        {"parameter_version": nodes["readout"].parameter_version}
    )


def test_readout_uses_declared_query_coordinates():
    events = stream(2)
    keys = all_keys(2)
    full = replay_module.full_replay(events, domain_builder=builder_for(keys), readout_grid=grid())
    fit = full.fits[0]
    transform = fit.transform
    assert transform is not None
    mean, variance = fit.predict(0, (3.0, 4.0))
    normalized = transform.normalize((3.0, 4.0))
    normalized_mean, normalized_variance = fit.predict(0, transform.denormalize(normalized))
    assert mean == normalized_mean
    assert variance == normalized_variance


def test_version_changes_trigger_recomputation():
    events = stream(2)
    keys = all_keys(2)
    base = builder_for(keys)(replay_module.consume(events)[0])[0]
    changed = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=base.evidence_source_ids,
        decision_version="dv2",
        mapping_version="mv1",
    )
    assert changed.support_version != base.support_version
    assert changed.content_digest != base.content_digest
    node = DependencyNode(
        node_id="fit",
        kind="structure_fit",
        parent_ids=("domain",),
        parameter_version="p1",
        support_version=base.support_version,
        mapping_version="mv1",
        calibration_version="cal1",
    )
    assert node.is_stale_against({"support_version": changed.support_version})
    assert not node.is_stale_against({"support_version": base.support_version})


def test_event_chunking_does_not_change_the_ledger():
    events = stream(3)
    one_shot, _ = replay_module.consume(events)
    chunked, _ = replay_module.consume(replay_module.EventStream(DATASET, events.events[:1]))
    for event in events.events[1:]:
        chunked.observe(event.frame)
    assert chunked.ledger_stats() == one_shot.ledger_stats()


def test_isolation_covers_declared_roles():
    """A retracted source must not be read through any declared role.

    The domain declares one evidence pair and one context source; the retraction
    removes the second evidence source and the context source, and the content of
    both is replaced.  The readout must not move, and the context role must never
    have entered the design in the first place.
    """
    events = replay_module.EventStream(
        DATASET,
        (
            observe_event([1.0, 3.0], 0),
            observe_event([777.0, 555.0], 1),
        ),
    )
    keys = all_keys(2)
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=[keys[0], keys[1]],
        context_source_ids=[keys[2]],
        decision_version="dv1",
        mapping_version="mv1",
    )
    base_fit = replay_module.fit_structure(domain, replay_module.consume(events)[0])
    assert base_fit.estimate.independent_observation_count == 2
    assert float(base_fit.estimate.theta[0]) == pytest.approx(2.0, abs=1e-12)
    assert base_fit.diagnostics["context_used_as_evidence"] is False

    retracted = retract_sources(retract_sources(domain, [keys[1]]), [keys[2]])
    result = replay_module.fixed_decision_replay(
        events,
        frozen_domains=[retracted],
        retracted_ids=[keys[1], keys[2]],
        replacement=ContentReplacement("garbage", {keys[1]: 999.0, keys[2]: 999.0}),
        readout_grid=grid(),
        frozen_decision_deps=("motion-from-frame-0",),
    )
    assert result.unchanged
    assert result.conditional_on == ("motion-from-frame-0",)
    assert result.manifest["unchanged"] is True
    # The context role survives the retraction of the evidence sibling and is not
    # silently reclassified as evidence.
    assert retracted.context_source_ids == ()
    assert retracted.evidence_source_ids == (keys[0],)

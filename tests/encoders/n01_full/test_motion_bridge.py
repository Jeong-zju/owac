"""P2 motion bridge tests: causal-window offsets and redacted-ledger rebuilds.

The motion model is an injected constant packed-flow stub (the estimator's own
documented injection point), so the tests exercise the real ``CorrespondenceEstimator``
geometry, tap derivation and validity rules without the SEA-RAFT weights.

What is asserted, and why a wrong version fails:

* window offsets are compared with a *direct* ``estimator.estimate`` call on the same
  two arrived frames, and the fusion output must differ from the identity run;
* a recording estimator sees only already-arrived frame ids, at every step;
* reset clears one member's correspondence without touching the other, and a camera
  that never arrived has no correspondence at all;
* the ledger builders recompute on the redacted view: replacing the retracted
  source's content changes nothing in the counterfactual path, the builder's view
  really has the sample zeroed/missing, and a *stale* correspondence built before
  retraction is refused instead of silently reused.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from owac.backends.raw_correspondence import CorrespondenceEstimator, RawSensorFrame
from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full import replay as replay_module
from owac.frameworks.n01_full.adapters import DbsrAdapterConfig, DbsrFeatureAdapter
from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.domains import make_domain
from owac.frameworks.n01_full.ledger import MISSING, ContentReplacement, MeasurementLedger
from owac.frameworks.n01_full.motion_builders import (
    LedgerCorrespondenceBuilder,
    LedgerNoiseModelBuilder,
    RebuiltCorrespondence,
    WindowMotionProvider,
)
from owac.frameworks.n01_full.statistics import (
    InterpolationSpec,
    align_interpolation_sources,
    fit_structure,
)
from owac.raw.n01_noise import (
    NoiseCalibration,
    NoiseProvenance,
    SharedBlackLevel,
    ShotReadParameters,
)

DATASET = "motion-bridge"
EPISODE = "e0"
CAMERA = "c0"
GRID = 4


# ---------------------------------------------------------------------------------------
# fixtures: injected packed-flow model, recording estimator, tiny adapter
# ---------------------------------------------------------------------------------------


class _ConstantFlow(nn.Module):
    """``PackedFlowModel`` stub: the same constant pull field for every pair."""

    def __init__(self, packed_shape: tuple[int, int], flow: tuple[float, float]) -> None:
        super().__init__()
        self.packed_shape = (int(packed_shape[0]), int(packed_shape[1]))
        self.flow = (float(flow[0]), float(flow[1]))
        self.calls = 0

    def forward(self, image1, image2, iters=None, flow_gt=None, test_mode=False):
        self.calls += 1
        batch = int(image1.shape[0])
        flow = torch.zeros(batch, 2, *self.packed_shape, dtype=torch.float32)
        flow[:, 0] = self.flow[0]
        flow[:, 1] = self.flow[1]
        return {"final": flow}


class _RecordingEstimator(CorrespondenceEstimator):
    """Records every ``(reference, source)`` frame pair the provider asks for."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pairs: list[tuple[int, int]] = []

    def estimate(self, reference, source, **kwargs):
        self.pairs.append((int(reference.frame_id), int(source.frame_id)))
        return super().estimate(reference, source, **kwargs)


def _estimator(model: nn.Module, mode: str = "rgb3") -> CorrespondenceEstimator:
    return CorrespondenceEstimator(mode=mode, model=model, iters=1, device="cpu")


def _provider(estimator: CorrespondenceEstimator, *, dataset_id: str = DATASET):
    return WindowMotionProvider(
        estimator, dataset_id=dataset_id, cfa_pattern="rgb3", channel_phase=()
    )


def _tiny_adapter(**overrides) -> DbsrFeatureAdapter:
    payload = {
        "stem_width": 4,
        "residual_blocks": 1,
        "feature_width": 8,
        "projection_width": 4,
        "offset_width": 8,
        "max_history": 4,
        "pool_grid": 2,
        "tokens_per_camera": 4,
        "readout_width": 16,
    }
    payload.update(overrides)
    return DbsrFeatureAdapter(DbsrAdapterConfig(**payload))


def _adapter_frame(values, *, camera="a", frame_ids=None, episodes=None, timestamps=None):
    batch = int(values.shape[0])
    return MeasurementFrame(
        values=values,
        valid_mask=torch.ones_like(values, dtype=torch.bool),
        frame_ids=torch.as_tensor([0] * batch if frame_ids is None else frame_ids),
        episode_ids=list(episodes) if episodes is not None else ["e0"] * batch,
        camera_id=camera,
        timestamp=torch.as_tensor(
            [0.0] * batch if timestamps is None else timestamps, dtype=torch.float32
        ),
    )


def _state_frame(state, camera: str, slot: int, member: int = 0) -> RawSensorFrame:
    """The same RawSensorFrame the provider builds, derived independently in the test."""
    values = state.cameras[camera].values[slot, member].float()
    valid = state.cameras[camera].valid[slot, member]
    return RawSensorFrame(
        dataset_id=DATASET,
        episode_id="e0",
        camera_id=camera,
        frame_id=int(state.cameras[camera].frame_ids[slot, member]),
        values=values.masked_fill(~valid, 0),
        channel_phase=(),
        cfa_pattern="rgb3",
        sensor_origin_xy=(0, 0),
        validity=valid,
        calibration_version="unversioned",
    )


# ---------------------------------------------------------------------------------------
# goal 1: the window provider really drives the adapter's fusion
# ---------------------------------------------------------------------------------------


def test_window_offsets_equal_direct_estimator_calls_and_change_the_fusion():
    torch.manual_seed(0)
    model = _ConstantFlow((8, 8), (0.25, 0.25))
    estimator = _estimator(model)
    adapter = _tiny_adapter()
    adapter.set_motion_provider(_provider(estimator))
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    frames = [
        torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(1 + step))
        for step in range(3)
    ]
    output = None
    for step, values in enumerate(frames):
        output = adapter.step(
            _adapter_frame(values, frame_ids=[step], timestamps=[float(step)]),
            None,
            state,
            budget,
        )
        state = output.state
    assert output.diagnostics["window_frames"] == 3
    assert output.diagnostics["p2_motion_status"] == "motion_provider"
    entry = state.correspondence_for("a")
    assert entry is not None
    assert entry.window_frames == output.diagnostics["window_frames"]
    assert tuple(entry.offsets_xy.shape) == (1, 3, 2, 8, 8)
    assert float(entry.offsets_xy[0, 0].abs().sum()) == 0.0
    assert entry.valid.tolist() == [[True, True, True]]
    assert entry.mapping_version == "n01-full-v1"
    assert entry.risk_kind == "unavailable"
    # Newest first: slot 0 is the reference frame 2, then 1, then 0.
    assert entry.source_frame_ids == (2, 1, 0)

    reference = _state_frame(state, "a", 0)
    for slot in (1, 2):
        direct = estimator.estimate(reference, _state_frame(state, "a", slot))
        assert torch.equal(entry.offsets_xy[0, slot], direct.packed_flow.to(torch.float32))

    # The offsets are real model input: an identity adapter (same weights) must differ.
    identity = _tiny_adapter()
    identity.load_state_dict(adapter.state_dict())
    identity_state = identity.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    identity_output = None
    for step, values in enumerate(frames):
        identity_output = identity.step(
            _adapter_frame(values, frame_ids=[step], timestamps=[float(step)]),
            None,
            identity_state,
            ComputeBudget(max_updates=1, max_history=identity.config.max_history),
        )
        identity_state = identity_output.state
    assert identity_state.correspondence_for("a").mapping_version == "identity"
    assert not torch.equal(adapter.readout(output).values, identity.readout(identity_output).values)


def test_provider_never_reads_a_future_frame_and_keeps_the_declared_window():
    recording = _RecordingEstimator(mode="rgb3", model=_ConstantFlow((8, 8), (0.0, 0.5)))
    adapter = _tiny_adapter()
    adapter.set_motion_provider(_provider(recording))
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    expected_pairs = 0
    for frame_id in range(5):
        values = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(frame_id))
        output = adapter.step(
            _adapter_frame(values, frame_ids=[frame_id], timestamps=[float(frame_id)]),
            None,
            state,
            budget,
        )
        state = output.state
        entry = state.correspondence_for("a")
        assert entry.window_frames == output.diagnostics["window_frames"] == min(frame_id + 1, 4)
        assert entry.offsets_xy.shape[1] == output.diagnostics["window_frames"]
        expected_pairs += max(min(frame_id + 1, 4) - 1, 0)
        assert len(recording.pairs) == expected_pairs
        for reference_id, source_id in recording.pairs:
            assert source_id <= frame_id, "a future exposure entered the estimator"
            assert reference_id == frame_id or reference_id < frame_id
    # Each step's pairs are the reference against strictly older sources, newest first.
    grouped: dict[int, list[int]] = {}
    for reference_id, source_id in recording.pairs:
        grouped.setdefault(reference_id, []).append(source_id)
    for reference_id, sources in grouped.items():
        assert sources == sorted(sources, reverse=True)
        assert all(source < reference_id for source in sources)


def test_no_provider_keeps_identity_offsets_and_the_old_contract():
    adapter = _tiny_adapter()
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    output = adapter.step(
        _adapter_frame(torch.randn(1, 3, 8, 8), frame_ids=[0]),
        None,
        state,
        ComputeBudget(max_updates=1, max_history=adapter.config.max_history),
    )
    entry = output.state.correspondence_for("a")
    assert entry.mapping_version == "identity"
    assert entry.risk_kind == "unavailable"
    assert output.diagnostics["p2_motion_status"] == "identity_pending_p2"
    assert float(entry.offsets_xy.abs().sum()) == 0.0


def test_motion_offsets_must_cover_the_step_window():
    adapter = _tiny_adapter()
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    for frame_id in range(2):
        output = adapter.step(
            _adapter_frame(torch.randn(1, 3, 8, 8), frame_ids=[frame_id]), None, state, budget
        )
        state = output.state
    with pytest.raises(ValueError, match="cannot change the window"):
        adapter.step(
            _adapter_frame(torch.randn(1, 3, 8, 8), frame_ids=[2]),
            None,
            state,
            budget,
            offsets_xy=torch.zeros(1, 2, 2, 8, 8),
        )


def test_reset_detach_and_missing_camera_keep_motion_per_camera():
    recording = _RecordingEstimator(mode="rgb3", model=_ConstantFlow((8, 8), (0.5, 0.0)))
    adapter = _tiny_adapter()
    adapter.set_motion_provider(_provider(recording))
    budget = ComputeBudget(max_updates=1, max_history=adapter.config.max_history)
    state = adapter.init_state(2, {"camera_ids": ["a", "b"], "shape": (8, 8)}, "cpu")

    def feed(state, camera, frame_ids, episodes):
        output = None
        values = torch.randn(
            len(frame_ids), 3, 8, 8, generator=torch.Generator().manual_seed(frame_ids[0] + 10)
        )
        output = adapter.step(
            _adapter_frame(
                values,
                camera=camera,
                frame_ids=frame_ids,
                episodes=episodes,
                timestamps=[float(item) for item in frame_ids],
            ),
            None,
            state,
            budget,
        )
        return output

    state = feed(state, "a", [0, 1], ["e0", "e1"]).state
    state = feed(state, "a", [2, 3], ["e0", "e1"]).state
    pairs_before = len(recording.pairs)
    state = feed(state, "b", [10, 11], ["e0", "e1"]).state
    output = feed(state, "b", [12, 13], ["e0", "e1"])
    state = output.state
    # Camera b's second arrival only ever paired its own two frames.
    assert recording.pairs[pairs_before:] == [(12, 10), (13, 11)]
    entry_b = state.correspondence_for("b")
    before_b = entry_b.offsets_xy.clone()
    assert state.correspondence_for("a").source_frame_ids == (2, 0)

    partial = adapter.reset(state, [True, False])
    entry_a = partial.correspondence_for("a")
    entry_b = partial.correspondence_for("b")
    assert not bool(entry_a.valid[0].any())
    assert float(entry_a.offsets_xy[0].abs().sum()) == 0.0
    assert bool(entry_b.valid[1].any())
    assert torch.equal(entry_b.offsets_xy[1:2], before_b[1:2])

    detached = adapter.detach_state(partial)
    assert detached.correspondence_for("b").offsets_xy.grad_fn is None

    cleared = adapter.reset(partial, None)
    assert cleared.correspondence == {}

    # A camera that never arrived has no correspondence entry at all.
    fresh = adapter.init_state(1, {"camera_ids": ["a", "b"], "shape": (8, 8)}, "cpu")
    after = feed(fresh, "b", [3], ["e0"]).state
    assert after.correspondence_for("a") is None
    assert after.correspondence_for("b") is not None


# ---------------------------------------------------------------------------------------
# goal 2: ledger builders recompute on the redacted view
# ---------------------------------------------------------------------------------------


def _calibration() -> NoiseCalibration:
    provenance = NoiseProvenance(
        kind="explicit_synthesis", split="synthesis", source="fixture", method="declared"
    )
    return NoiseCalibration(
        calibration_version="fixture-v1",
        units="dn",
        shot_read=ShotReadParameters(shot=0.5, read_var=2.0, units="dn", provenance=provenance),
        black_level=SharedBlackLevel(
            kind="per_frame",
            id_template="{dataset_id}:{episode_id}:{camera_id}:{frame_id}",
            variance=0.25,
            provenance=provenance,
        ),
    )


def _packed_values(frame_id: int, *, spike: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(100 + frame_id)
    values = torch.randn(3, GRID, GRID, generator=generator)
    if frame_id == 0:
        values[0, 2, 2] = spike
    return values


def _observe(values: torch.Tensor, frame_id: int):
    return replay_module.ObserveEvent(
        MeasurementFrame(
            values=values[None],
            frame_ids=[frame_id],
            episode_ids=[EPISODE],
            camera_id=CAMERA,
            timestamp=torch.tensor([float(frame_id)]),
            new_measurement=torch.tensor([True]),
        )
    )


def _stream(spike: float = 1.0):
    return replay_module.EventStream(
        DATASET, tuple(_observe(_packed_values(frame, spike=spike), frame) for frame in range(3))
    )


def _all_keys(frames: int = 3) -> list[SourceKey]:
    return [
        SourceKey(DATASET, EPISODE, CAMERA, frame, channel, y, x)
        for frame in range(frames)
        for channel in range(3)
        for y in range(GRID)
        for x in range(GRID)
    ]


def _stream_frames(frames: int, *, spike: float = 1.0):
    return replay_module.EventStream(
        DATASET,
        tuple(_observe(_packed_values(frame, spike=spike), frame) for frame in range(frames)),
    )


def _domain_builder(keys):
    def builder(ledger: MeasurementLedger):
        evidence = [
            key for key in keys if ledger.has_source(key) and ledger.validity_of(key) == "valid"
        ]
        return [
            make_domain(
                "D",
                reference_chart="ref",
                basis="phase_local_constant",
                phase=None,
                evidence_source_ids=evidence,
                decision_version="dv1",
                mapping_version="mv1",
            )
        ]

    return builder


def _builders(keys):
    estimator = _estimator(_ConstantFlow((GRID, GRID), (0.25, 0.25)))
    correspondence = LedgerCorrespondenceBuilder(
        estimator, evidence_ids=keys, cfa_pattern="rgb3", camera_id=CAMERA
    )
    noise = LedgerNoiseModelBuilder(_calibration(), tuple(keys))
    return correspondence, noise


def _grid():
    return replay_module.ReadoutGrid(
        (
            replay_module.QueryPoint("q0", "D", 0, 2.0, 2.0),
            replay_module.QueryPoint("q1", "D", 1, 3.0, 2.0),
        )
    )


def test_rebuilt_correspondence_rows_taps_and_fit_consumption():
    keys = _all_keys()
    correspondence, noise_builder = _builders(keys)
    ledger, _receipts = replay_module.consume(_stream())
    rebuilt = correspondence(ledger)
    assert isinstance(rebuilt, RebuiltCorrespondence)
    columns = correspondence.usable_columns(ledger)
    assert rebuilt.source_ids == tuple(columns)
    assert rebuilt.weights.shape[1] == len(columns)
    assert rebuilt.valid_rows > 0
    assert rebuilt.dropped_rows > 0  # the frame border is dropped, never filled
    sums = rebuilt.weights.sum(dim=1)
    assert float(sums.min()) > 1 - 1e-12 and float(sums.max()) < 1 + 1e-12
    phases = [key.channel_or_phase for key in rebuilt.source_ids]
    for row in rebuilt.weights:
        touched = {phases[index] for index in row.nonzero().reshape(-1).tolist()}
        assert len(touched) == 1, "an interpolated row mixes phases"
    assert rebuilt.risk_kind == "unavailable" and rebuilt.risk is None
    assert rebuilt.describe()["weights_sha256"] == rebuilt.describe()["weights_sha256"]

    domain = _domain_builder(keys)(ledger)[0]
    noise = noise_builder(ledger)
    fit = fit_structure(domain, ledger, rebuilt, noise)
    assert fit.diagnostics["interpolated"] is True
    assert fit.diagnostics["correspondence_report"]["kind"] == "rebuilt-correspondence"
    assert fit.diagnostics["interpolation_sources_aligned"] == len(columns)
    assert fit.blocks

    # A named column that the domain does not carry is refused, not silently dropped.
    extra = SourceKey(DATASET, EPISODE, "other", 0, 0, 0, 0)
    spec = InterpolationSpec(
        row_positions=((0.0, 0.0),),
        weights=torch.tensor([[1.0]]),
        source_ids=(extra,),
    )
    with pytest.raises(ValueError, match="outside the domain's usable evidence set"):
        align_interpolation_sources(spec, columns)
    aligned = align_interpolation_sources(
        InterpolationSpec(
            row_positions=((0.0, 0.0),),
            weights=torch.tensor([[1.0]]),
            source_ids=(columns[0],),
        ),
        [columns[1], columns[0]],
    )
    assert float(aligned.weights[0, 1]) == 1.0 and float(aligned.weights[0, 0]) == 0.0


def test_counterfactual_result_is_invariant_under_replaced_retracted_content():
    keys = _all_keys()
    correspondence, noise_builder = _builders(keys)
    retracted = SourceKey(DATASET, EPISODE, CAMERA, 0, 0, 2, 2)

    def run(stream):
        return replay_module.counterfactual_replay(
            stream,
            domain_builder=_domain_builder(keys),
            retracted_ids=[retracted],
            readout_grid=_grid(),
            correspondence_builder=correspondence,
            noise_model_builder=noise_builder,
        )

    first = run(_stream(spike=1.0))
    second = run(_stream(spike=50.0))
    assert first.claim == "exact_independence_in_event_stream"
    assert first.manifest["correspondence_recomputed"] is True
    assert first.manifest["noise_recomputed"] is True
    assert torch.equal(first.result.readout.means, second.result.readout.means), (
        "the retracted content still reaches the counterfactual result"
    )
    assert torch.equal(first.result.snapshot.representation, second.result.snapshot.representation)

    # Positive control: retraction really matters, so the invariance above is not vacuous.
    full = replay_module.full_replay(
        _stream(spike=1.0),
        domain_builder=_domain_builder(keys),
        readout_grid=_grid(),
    )
    assert not torch.equal(full.readout.means, first.result.readout.means)

    # A stale correspondence built before the retraction is refused, not reused.
    ledger, _ = replay_module.consume(_stream(spike=1.0))
    stale = correspondence(ledger)
    with pytest.raises(ValueError, match="outside the domain's usable evidence set"):
        replay_module.counterfactual_replay(
            _stream(spike=1.0),
            domain_builder=_domain_builder(keys),
            retracted_ids=[retracted],
            readout_grid=_grid(),
            correspondence=stale,
        )


def test_builder_receives_the_cleared_view_and_only_uses_retained_raw():
    keys = _all_keys()
    correspondence, _noise = _builders(keys)
    retracted = SourceKey(DATASET, EPISODE, CAMERA, 0, 0, 2, 2)
    seen: list[MeasurementLedger] = []

    def probe(view: MeasurementLedger):
        seen.append(view)
        return correspondence(view)

    replay_module.counterfactual_replay(
        _stream(spike=7.0),
        domain_builder=_domain_builder(keys),
        retracted_ids=[retracted],
        readout_grid=_grid(),
        correspondence_builder=probe,
    )
    assert len(seen) == 1
    view = seen[0]
    assert view.has_source(retracted) is False
    frame = view.get_frame(EPISODE, CAMERA, 0)
    assert frame is not None
    assert float(frame.values[0, 2, 2]) == 0.0
    assert int(frame.validity[0, 2, 2]) == MISSING
    rebuilt = correspondence(view)
    assert retracted not in rebuilt.source_ids
    # Every column the builder emits is a real, valid source of the very view it saw.
    assert set(rebuilt.source_ids) == set(correspondence.usable_columns(view))
    assert all(view.has_source(key) for key in rebuilt.source_ids)
    assert all(view.validity_of(key) == "valid" for key in rebuilt.source_ids)


def test_noise_builder_recomputes_from_retained_means_only():
    keys = _all_keys()
    _correspondence, noise_builder = _builders(keys)
    retracted = SourceKey(DATASET, EPISODE, CAMERA, 0, 0, 2, 2)
    captured: list[tuple[MeasurementLedger, object]] = []

    def probe(view: MeasurementLedger):
        model = noise_builder(view)
        captured.append((view, model))
        return model

    replay_module.counterfactual_replay(
        _stream(spike=9.0),
        domain_builder=_domain_builder(keys),
        retracted_ids=[retracted],
        readout_grid=_grid(),
        correspondence_builder=_builders(keys)[0],
        noise_model_builder=probe,
    )
    view, model = captured[0]
    assert model.version == _calibration().noise_version
    identities = {key.identity() for key in model.source_order}
    assert retracted.identity() not in identities
    assert len(model.source_order) == len(keys) - 1
    assert all(view.has_source(key) for key in model.source_order)
    assert all(view.validity_of(key) == "valid" for key in model.source_order)
    # The declared law is used: variance = shot * max(mean - black, 0) + read + block.
    means = [float(view.get_source(key).raw_value) for key in model.source_order]
    expected = [
        _calibration().shot_read.variance([mean], black_level=0.0)[0].item() + 0.25
        for mean in means
    ]
    assert model.variances == pytest.approx(expected)


def test_fixed_decision_replay_is_unchanged_only_with_frozen_post_retraction_decisions():
    keys = _all_keys()
    correspondence, noise_builder = _builders(keys)
    retracted = SourceKey(DATASET, EPISODE, CAMERA, 0, 0, 2, 2)
    ledger, _ = replay_module.consume(_stream(spike=3.0))
    view = ledger.without([retracted])
    domains = _domain_builder(keys)(view)
    spec = correspondence(view)
    noise = noise_builder(view)
    replacement = ContentReplacement("replacement-1", {retracted: 123.0})

    result = replay_module.fixed_decision_replay(
        _stream(spike=3.0),
        frozen_domains=domains,
        retracted_ids=[retracted],
        replacement=replacement,
        correspondence=spec,
        noise_model=noise,
        readout_grid=_grid(),
    )
    assert result.max_abs_diff == 0.0
    assert result.unchanged is True

    # The other direction: a decision that still carries the retracted source is refused.
    stale = correspondence(ledger)
    with pytest.raises(ValueError, match="outside the domain's usable evidence set"):
        replay_module.fixed_decision_replay(
            _stream(spike=3.0),
            frozen_domains=domains,
            retracted_ids=[retracted],
            replacement=replacement,
            correspondence=stale,
            noise_model=noise,
            readout_grid=_grid(),
        )


def test_provider_survives_deepcopy_for_state_dict_transfers():
    adapter = _tiny_adapter()
    adapter.set_motion_provider(_provider(_estimator(_ConstantFlow((8, 8), (0.0, 0.0)))))
    clone = copy.deepcopy(adapter)
    assert clone.motion_provider is not None
    assert float(clone.state_dict()["backbone.upstream.init_layer.0.weight"].abs().sum()) > 0


def test_builder_view_is_bounded_to_the_exact_replay_window():
    """Nothing beyond the retained 32-frame window may reach the builder."""
    total = 36
    keys = _all_keys(total)
    correspondence = LedgerCorrespondenceBuilder(
        _estimator(_ConstantFlow((GRID, GRID), (0.25, 0.25))),
        evidence_ids=keys,
        cfa_pattern="rgb3",
        camera_id=CAMERA,
    )
    seen: list[MeasurementLedger] = []

    def probe(view: MeasurementLedger):
        seen.append(view)
        return correspondence(view)

    retracted = SourceKey(DATASET, EPISODE, CAMERA, total - 1, 0, 1, 1)
    result = replay_module.counterfactual_replay(
        _stream_frames(total),
        domain_builder=_domain_builder(keys),
        retracted_ids=[retracted],
        readout_grid=_grid(),
        correspondence_builder=probe,
    )
    view = seen[0]
    assert view.unique_frame_count() == 32
    assert not view.has_source(SourceKey(DATASET, EPISODE, CAMERA, 0, 0, 0, 0))
    assert all(int(frame.identity.frame_id) >= total - 32 for frame in view.frames()), (
        "a frame outside the exact window entered the builder's input"
    )
    rebuilt = correspondence(view)
    assert min(int(key.frame_id) for key in rebuilt.source_ids) >= total - 32
    assert result.result.history_loss  # the dropped frames are recorded, never summarised away

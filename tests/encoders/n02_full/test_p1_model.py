"""P1 model tests: minimal causal step, evidence accounting and no second likelihood.

The T1/T2 modules land in the same batch, so the encoder and sensor model are
same-signature fakes here; the step, the ledger accounting and the objective
identity are the real code under test. The integration check against the real
modules runs separately.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from owac.core.encoder_contract_n02 import ComputeBudget, EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.factors import (
    CandidateDictionary,
    CandidateGeometry,
    CandidateSpec,
    HistoryContext,
    ObservationModel,
    QueryAlignmentError,
    RawQuery,
)
from owac.frameworks.n02_full.model import ObjectiveRegister, step_bundle
from owac.frameworks.n02_full.state import (
    FeatureLevel,
    MeasurementKey,
    MeasurementLedger,
    ObservationBundle,
    SequenceState,
    TrainingTargets,
    measurement_set_hash,
)

FEATURE_DIM = 8
WORLD_SIZE = 2
CAMERA = "cam0"
SENSOR_SHAPE = {CAMERA: (4, 4)}


# --------------------------------------------------------------------------------------
# same-signature fakes
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class FakeRawPrediction:
    mean: torch.Tensor
    variance: torch.Tensor
    masks: dict
    units: str


@dataclass(slots=True)
class FakeMeasurementScore:
    relative_terms: dict
    raw_nll_or_energy: torch.Tensor
    local_stats: dict
    kind: str
    per_valid_scalar: torch.Tensor


@dataclass(slots=True)
class FakeInterferenceEstimate:
    log_exposure: torch.Tensor
    camera_translation: torch.Tensor
    noise_scale: torch.Tensor
    source: str = "history"


class FakeSensorModel:
    raw_prediction_type = FakeRawPrediction

    def __init__(self, kind: str = "proper_nll") -> None:
        self.kind = kind
        self.score_calls = 0

    def score(self, prediction: FakeRawPrediction, raw_query: RawQuery) -> FakeMeasurementScore:
        self.score_calls += 1
        values = raw_query.values
        mean = prediction.mean
        variance = prediction.variance
        nll = 0.5 * (
            (values - mean) ** 2 / variance + torch.log(variance) + math.log(2.0 * math.pi)
        )
        return FakeMeasurementScore(
            relative_terms={"variance": variance.clone()},
            raw_nll_or_energy=nll,
            local_stats={},
            kind=self.kind,
            per_valid_scalar=nll,
        )


class FakeEncoder:
    def __init__(self, feature_dim: int = FEATURE_DIM, value: float = 2.0) -> None:
        self.feature_dim = feature_dim
        self.value = float(value)
        self.calls = 0

    def encode(self, bundle) -> SimpleNamespace:
        self.calls += 1
        batch = len(bundle.episode_ids) or 1
        level = FeatureLevel(
            values=torch.full((batch, self.feature_dim, 2, 2), self.value),
            valid=torch.ones((batch, 2, 2), dtype=torch.bool),
            sensor_coordinates=torch.zeros((batch, 2, 2, 2)),
            support_ids=(),
            level_id="stride8",
        )
        cameras = tuple(bundle.frames)
        return SimpleNamespace(
            levels={"stride8": level},
            camera_ids=cameras,
            per_camera=lambda: {camera: {"stride8": level} for camera in cameras},
            input_mode="raw3",
            sensor_shape={},
            support_hash="fake",
        )


def make_config(**overrides) -> SimpleNamespace:
    values = {
        "feature_dim": FEATURE_DIM,
        "candidate_world_size": WORLD_SIZE,
        "candidate_embedding_dim": 8,
        "decoder_grid": 4,
        "decoder_channels": 4,
        "decoder_hidden": 8,
        "sensor_shape": dict(SENSOR_SHAPE),
        "structure_version": "p1-test-structure",
        "initial_prior_version": "prior-v1",
        "query_stride": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_model(
    *, encoder=None, sensor_model=None, dictionary=None, **config_overrides
) -> ObservationModel:
    torch.manual_seed(1234)
    model = ObservationModel(
        encoder if encoder is not None else FakeEncoder(),
        sensor_model if sensor_model is not None else FakeSensorModel(),
        make_config(**config_overrides),
        version="p1-test-model",
    )
    if dictionary is not None:
        model.bind_dictionary(dictionary)
    return model


# --------------------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------------------

TRUTH = torch.tensor([[1.0, 1.0], [3.0, 2.0]], dtype=torch.float32)


def geometry(positions) -> CandidateGeometry:
    return CandidateGeometry(world_px=torch.as_tensor(positions, dtype=torch.float32))


def make_dictionary(*, all_wrong: bool = False, correct_index: int = 0) -> CandidateDictionary:
    wrongs = []
    for index, offset in enumerate(((1.0, 0.0), (0.0, -2.0))):
        shifted = TRUTH.clone()
        shifted[index] += torch.as_tensor(offset, dtype=torch.float32)
        wrongs.append(CandidateSpec(f"wrong-{index:02d}", "wrong", geometry(shifted)))
    if all_wrong:
        return CandidateDictionary(specs=tuple(wrongs), all_wrong=True)
    correct = CandidateSpec("correct", "correct", geometry(TRUTH.clone()))
    specs = tuple(wrongs[:correct_index]) + (correct,) + tuple(wrongs[correct_index:])
    return CandidateDictionary(specs=specs)


def make_history(
    *,
    batch: int = 1,
    last_frame_index: int | None = 1,
    feature: float = 0.5,
    eta=None,
    eta_fit_keys=(),
) -> HistoryContext:
    features = torch.full((batch, FEATURE_DIM), feature)
    level = FeatureLevel(
        values=features.unsqueeze(-1).unsqueeze(-1),
        valid=torch.ones((batch, 1, 1), dtype=torch.bool),
        sensor_coordinates=torch.zeros((batch, 1, 1, 2)),
        support_ids=(),
        level_id="stride8",
    )
    return HistoryContext(
        pooled_features=features,
        levels={"stride8": level},
        episode_ids=("ep0",) * batch,
        anchor_ids=("ep0#anchor",) * batch,
        last_frame_index=last_frame_index,
        eta=eta,
        eta_fit_keys=eta_fit_keys,
    )


def make_frame(
    *,
    values=None,
    frame_id: str = "cam0-f0002",
    episode: str = "ep0",
    calibration_version: str | None = None,
    acquisition_id: str | None = None,
    new_measurement=True,
    unit: str = "dn_12bit",
    camera: str = CAMERA,
    calibration: dict | None = None,
) -> MeasurementFrame:
    height, width = 4, 4
    if values is None:
        values = torch.arange(height * width, dtype=torch.float32).reshape(height, width)
    tensor = torch.as_tensor(values)
    mask_shape = tuple(tensor.shape[:-2]) + tuple(tensor.shape[-2:])
    metadata = {"unit": unit}
    if calibration_version is not None:
        metadata["calibration_version"] = calibration_version
    if acquisition_id is not None:
        metadata["acquisition_id"] = acquisition_id
    if calibration is not None:
        metadata["calibration"] = dict(calibration)
    return MeasurementFrame(
        values=tensor,
        valid_mask=torch.ones(mask_shape, dtype=torch.bool),
        bad_mask=torch.zeros(mask_shape, dtype=torch.bool),
        saturated_mask=torch.zeros(mask_shape, dtype=torch.bool),
        episode_ids=(episode,),
        frame_ids=(frame_id,),
        camera_id=camera,
        new_measurement=new_measurement,
        metadata=metadata,
    )


def single_bundle(
    *,
    episode: str = "ep0",
    frame_id: str = "cam0-f0002",
    new_time: bool = True,
    values=None,
    calibration_version: str | None = None,
    unit: str = "dn_12bit",
    calibration: dict | None = None,
) -> ObservationBundle:
    frame = make_frame(
        values=values,
        frame_id=frame_id,
        episode=episode,
        calibration_version=calibration_version,
        new_measurement=new_time,
        unit=unit,
        calibration=calibration,
    )
    return ObservationBundle(
        frames={CAMERA: frame},
        context=EmbodimentContext(),
        episode_ids=(episode,),
        anchor_ids=(f"{episode}#anchor-f0002",),
        new_time_mask=torch.tensor([new_time]),
    )


def batch_bundle(
    *,
    episodes=("ep-a", "ep-b"),
    frame_ids=("cam0-f0001", "cam0-f0002"),
    new_time=(True, True),
    rows=None,
    calibration_version: str | None = None,
) -> ObservationBundle:
    table = (
        torch.stack([torch.full((4, 4), float(index + 1)) for index in range(len(episodes))])
        if rows is None
        else torch.stack([torch.as_tensor(row, dtype=torch.float32) for row in rows])
    )
    frame = make_frame(
        values=table,
        frame_id=frame_ids[0],
        episode=episodes[0],
        calibration_version=calibration_version,
    )
    frame.episode_ids = tuple(episodes)
    frame.frame_ids = tuple(frame_ids)
    frame.new_measurement = torch.tensor(list(new_time))
    return ObservationBundle(
        frames={CAMERA: frame},
        context=EmbodimentContext(),
        episode_ids=tuple(episodes),
        anchor_ids=tuple(f"{episode}#anchor" for episode in episodes),
        new_time_mask=torch.tensor(list(new_time)),
    )


# --------------------------------------------------------------------------------------
# registration, locking, evidence
# --------------------------------------------------------------------------------------


def test_step_registers_keys_and_locks_prequential_before_fit() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    output = step_bundle(single_bundle(), SequenceState(), model, make_history(), ledger=ledger)

    assert output.evidence_count == 1
    assert ledger.state_version == 1
    assert output.cost["new_evidence"] == 1.0
    assert output.cost["predictions"] == 3.0
    assert output.cost["locked_checks"] == 3.0
    assert output.cost["fit_scores"] == 3.0
    assert output.cost["likelihood_terms"] == 1.0
    assert output.cost["duplicate_objectives"] == 0.0
    assert output.state.episode_ids == ("ep0",)
    assert output.objective.measurement_set_hash == measurement_set_hash(ledger.keys())
    assert output.objective.model_version == "p1-test-model"
    assert output.objective.structure_version == "p1-test-structure"
    assert output.objective.initial_prior_version == "prior-v1"

    # the locked prequential check and the fit score are two different raw quantities
    assert not torch.equal(
        output.locked_checks[0].raw_nll_or_energy, output.fit_scores[0].raw_nll_or_energy
    )
    assert len(output.locked_checks) == len(output.worlds) == len(output.fit_scores)


def test_new_time_false_does_not_predict_but_still_registers() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    output = step_bundle(
        single_bundle(new_time=False), SequenceState(), model, make_history(), ledger=ledger
    )
    assert output.cost["predictions"] == 0.0
    assert output.locked_checks == ()
    assert output.fit_scores == ()
    assert output.worlds and output.best_world is None
    assert output.evidence_count == 1


def test_retransmission_adds_no_evidence_and_no_second_likelihood() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    bundle = single_bundle()
    first = step_bundle(bundle, SequenceState(), model, make_history(), ledger=ledger)
    second = step_bundle(bundle, first.state, model, make_history(), ledger=ledger)

    assert second.evidence_count == first.evidence_count == 1
    assert second.cost["new_evidence"] == 0.0
    assert ledger.state_version == 1
    assert second.cost["likelihood_terms"] == first.cost["likelihood_terms"] == 1.0
    assert second.cost["duplicate_objectives"] == 1.0
    assert second.objective == first.objective
    for left, right in zip(first.locked_checks, second.locked_checks, strict=True):
        assert torch.equal(left.raw_nll_or_energy, right.raw_nll_or_energy)


def test_calibration_change_only_replaces_the_derived_version() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    first = step_bundle(single_bundle(), SequenceState(), model, make_history(), ledger=ledger)
    changed = step_bundle(
        single_bundle(calibration_version="cal-2"),
        first.state,
        model,
        make_history(),
        ledger=ledger,
    )
    assert ledger.evidence_count == 1
    assert ledger.state_version == 2  # the calibration replacement, not a new measurement
    assert ledger.records()[0].calibration_version == "cal-2"
    assert changed.cost["new_evidence"] == 0.0
    assert changed.cost["duplicate_objectives"] == 0.0
    assert changed.cost["likelihood_terms"] == 2.0
    assert changed.objective != first.objective
    assert changed.objective.calibration_versions == ("cal-2",)


def test_duplicate_acquisition_ids_are_one_measurement() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    bundle = single_bundle()
    bundle.frames[CAMERA].metadata["acquisition_id"] = "acq-7"
    step_bundle(bundle, SequenceState(), model, make_history(), ledger=ledger)
    assert ledger.evidence_count == 1
    assert ledger.records()[0].key.acquisition_id == "acq-7:cam0-f0002"


# --------------------------------------------------------------------------------------
# causality at the step level
# --------------------------------------------------------------------------------------


def test_current_frame_values_do_not_change_the_locked_prediction() -> None:
    model = make_model(dictionary=make_dictionary())
    low = torch.zeros(4, 4)
    high = torch.full((4, 4), 500.0)
    first = step_bundle(
        single_bundle(values=low),
        SequenceState(),
        model,
        make_history(),
        ledger=MeasurementLedger(),
    )
    second = step_bundle(
        single_bundle(values=high),
        SequenceState(),
        model,
        make_history(),
        ledger=MeasurementLedger(),
    )
    for left, right in zip(first.locked_checks, second.locked_checks, strict=True):
        # prediction-only terms are bit identical; the check itself legitimately uses y_t
        assert torch.equal(left.relative_terms["variance"], right.relative_terms["variance"])
        assert not torch.equal(left.raw_nll_or_energy, right.raw_nll_or_energy)


def test_future_rows_and_targets_do_not_change_earlier_locked_checks() -> None:
    model = make_model(dictionary=make_dictionary())
    history = make_history(batch=2, last_frame_index=0)
    row0 = torch.zeros(4, 4)

    def run(row1, targets):
        bundle = batch_bundle(rows=[row0, row1], new_time=(True, True))
        return step_bundle(
            bundle, SequenceState(), model, history, ledger=MeasurementLedger(), targets=targets
        )

    noise = TrainingTargets(actions=torch.zeros(8, 2), action_mask=torch.ones(8, dtype=torch.bool))
    garbage = TrainingTargets(
        actions=torch.full((8, 2), 999.0), action_mask=torch.ones(8, dtype=torch.bool)
    )
    first = run(torch.ones(4, 4), noise)
    second = run(torch.full((4, 4), 250.0), garbage)
    assert first.cost["targets_present"] == second.cost["targets_present"] == 1.0
    for left, right in zip(first.locked_checks, second.locked_checks, strict=True):
        assert torch.equal(left.relative_terms["variance"][0], right.relative_terms["variance"][0])
        assert torch.equal(left.raw_nll_or_energy[0], right.raw_nll_or_energy[0])


def test_episode_reset_leaves_other_rows_bit_identical() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    state = SequenceState(
        episode_ids=("ep-a", "ep-b"),
        slot_hidden=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        existence=torch.tensor([[True], [True]]),
        geometry=torch.randn(2, 2, 2),
        version=4,
    )
    untouched_hidden = state.slot_hidden[1].clone()
    untouched_geometry = state.geometry[1].clone()
    history = make_history(batch=2, last_frame_index=0)
    bundle = batch_bundle(episodes=("ep-c", "ep-b"), new_time=(True, False))
    output = step_bundle(bundle, state, model, history, ledger=ledger)

    assert output.state.episode_ids == ("ep-c", "ep-b")
    assert output.state.version == 5
    assert torch.equal(output.state.slot_hidden[0], torch.zeros(2))
    assert torch.equal(output.state.slot_hidden[1], untouched_hidden)
    assert torch.equal(output.state.geometry[1], untouched_geometry)
    # the input state is never mutated in place
    assert state.episode_ids == ("ep-a", "ep-b")
    assert ledger.evidence_count == 2


def test_objective_is_scoped_to_the_episode_of_the_last_new_row() -> None:
    model = make_model(dictionary=make_dictionary())
    ledger = MeasurementLedger()
    history = make_history(batch=2, last_frame_index=0)
    output = step_bundle(
        batch_bundle(episodes=("ep-a", "ep-b")), SequenceState(), model, history, ledger=ledger
    )
    episode_keys = [record.key for record in ledger.records() if record.key.episode_id == "ep-b"]
    assert output.objective.measurement_set_hash == measurement_set_hash(episode_keys)
    assert len(episode_keys) == 1


# --------------------------------------------------------------------------------------
# calibration-only / empirical marking and oracle eta
# --------------------------------------------------------------------------------------


def test_first_frame_checks_are_marked_calibration_only() -> None:
    model = make_model(dictionary=make_dictionary())
    output = step_bundle(
        single_bundle(frame_id="cam0-f0001"),
        SequenceState(),
        model,
        make_history(last_frame_index=None),
        ledger=MeasurementLedger(),
    )
    assert output.cost["calibration_only_checks"] == 3.0
    for check in output.locked_checks:
        assert bool(check.local_stats["calibration_only"])
        assert check.kind == "empirical_energy"


def test_eta_fitted_on_the_query_marks_checks_empirical() -> None:
    model = make_model(dictionary=make_dictionary())
    bundle = single_bundle()
    key = MeasurementKey("ep0", CAMERA, "cam0-f0002", "cam0-f0002")
    eta = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([1.0]),
        source="history",
    )
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(eta=eta, eta_fit_keys=(key,)),
        ledger=MeasurementLedger(),
    )
    assert output.cost["empirical_checks"] == 3.0
    for check in output.locked_checks:
        assert bool(check.local_stats["empirical_check"])


def test_oracle_eta_is_refused_in_the_online_step() -> None:
    model = make_model(dictionary=make_dictionary())
    eta = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([1.0]),
        source="oracle",
    )
    with pytest.raises(ValueError, match="oracle eta"):
        step_bundle(
            single_bundle(),
            SequenceState(),
            model,
            make_history(eta=eta),
            ledger=MeasurementLedger(),
        )


# --------------------------------------------------------------------------------------
# world scoring, objective register, budget
# --------------------------------------------------------------------------------------


def test_best_world_is_the_fit_argmin_and_score_kinds_are_declared() -> None:
    model = make_model(dictionary=make_dictionary())
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    means = [float(score.raw_nll_or_energy.detach().mean()) for score in output.fit_scores]
    assert output.best_world is not None
    assert output.best_world.score == pytest.approx(min(means), rel=1e-6)
    for world, score in zip(output.worlds, output.fit_scores, strict=True):
        assert world.score == pytest.approx(
            float(score.raw_nll_or_energy.detach().mean()), rel=1e-6
        )
        assert world.score_kind in {"normalized_density", "empirical_energy", "importance_weight"}
    assert output.worlds[0].score_kind == "normalized_density"


def test_unknown_sensor_score_kind_is_refused() -> None:
    model = make_model(
        dictionary=make_dictionary(), sensor_model=FakeSensorModel(kind="mystery_energy")
    )
    with pytest.raises(ValueError, match="unknown score kind"):
        step_bundle(
            single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
        )


def test_objective_register_never_stores_two_entries_for_one_identity() -> None:
    register = ObjectiveRegister()
    ledger = MeasurementLedger()
    model = make_model(dictionary=make_dictionary())
    output = step_bundle(single_bundle(), SequenceState(), model, make_history(), ledger=ledger)
    first_new = register.add(output.objective, 1.0)
    second_new = register.add(output.objective, 9.0)
    assert first_new == (True, 1.0)
    assert second_new == (False, 1.0)
    assert len(register) == 1
    assert register.value(output.objective) == 1.0


def test_zero_update_budget_is_refused() -> None:
    model = make_model(dictionary=make_dictionary())
    with pytest.raises(ValueError, match="max_updates"):
        step_bundle(
            single_bundle(),
            SequenceState(),
            model,
            make_history(),
            ledger=MeasurementLedger(),
            budget=ComputeBudget(max_updates=0),
        )


def test_step_without_candidates_still_registers_and_is_honest() -> None:
    model = make_model()  # no dictionary bound
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    assert output.worlds == ()
    assert output.best_world is None
    assert output.locked_checks == ()
    assert output.evidence_count == 1


def test_candidates_are_not_refit_by_targets_and_geometry_stays_view_2d() -> None:
    model = make_model(dictionary=make_dictionary())
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    for world in output.worlds:
        assert world.entities[0].geometry_frame == "view_2d"
        assert set(world.entities[0].geometry) == {"world_px", "scale_px", "depth"}


# --------------------------------------------------------------------------------------
# integration with the delivered T2 sensor module
# --------------------------------------------------------------------------------------


def test_real_sensor_model_produces_real_prediction_and_score_types() -> None:
    """The step consumes ``sensor.SensorModel`` and returns sensor.py score objects."""
    from owac.frameworks.n02_full import sensor as sensor_mod

    sensor_model = sensor_mod.SensorModel(units="dn_12bit", density_verified=True)
    encoder = ParametricFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary(), sensor_model=sensor_model)
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    assert isinstance(output.locked_checks[0], sensor_mod.MeasurementScore)
    assert output.locked_checks[0].kind == "proper_nll"
    assert output.fit_scores[0].kind == "proper_nll"
    assert output.worlds[0].score_kind == "normalized_density"

    prediction = model.predict(
        output.worlds[0],
        make_history(),
        _query_spec_from_bundle(single_bundle()),
        call="prequential_check",
    )
    assert isinstance(prediction, sensor_mod.RawPrediction)
    assert prediction.units == "dn_12bit"
    assert bool((prediction.variance > 0).all())
    assert set(prediction.variance_components) == {"sensor", "model", "total"}

    # the real proper score stays differentiable back to the raw encoder
    assert output.world_scores.requires_grad
    output.world_scores.sum().backward()
    assert encoder.scale.grad is not None
    assert bool(torch.isfinite(encoder.scale.grad).all())
    assert float(encoder.scale.grad.detach().abs()) > 0.0


def _query_spec_from_bundle(bundle: ObservationBundle):
    from owac.frameworks.n02_full.model import _build_row_query

    row_query = _build_row_query(
        bundle, row=0, batch=1, episode_id="ep0", history=make_history(), stride=2
    )
    return row_query.spec


# --------------------------------------------------------------------------------------
# fix round 1: gradient path, differentiable world scores, per-camera semantics
# --------------------------------------------------------------------------------------


class ParametricFakeEncoder(torch.nn.Module):
    """A raw encoder whose feature value is a real parameter (gradient check)."""

    def __init__(self, feature_dim: int = FEATURE_DIM, value: float = 2.0) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = torch.nn.Parameter(torch.tensor(float(value)))
        self.calls = 0

    def encode(self, bundle) -> SimpleNamespace:
        self.calls += 1
        batch = len(bundle.episode_ids) or 1
        level = FeatureLevel(
            values=self.scale * torch.ones(batch, self.feature_dim, 2, 2),
            valid=torch.ones(batch, 2, 2, dtype=torch.bool),
            sensor_coordinates=torch.zeros(batch, 2, 2, 2),
            support_ids=(),
            level_id="stride8",
        )
        cameras = tuple(bundle.frames)
        return SimpleNamespace(
            levels={"stride8": level},
            camera_ids=cameras,
            per_camera=lambda: {camera: {"stride8": level} for camera in cameras},
            input_mode="raw3",
            sensor_shape={},
            support_hash="fake-parametric",
        )


def test_fit_scores_keep_gradient_to_the_raw_encoder() -> None:
    """P1 §5 Base lr 3e-5 needs the encoder graph: fit_scores must reach Base params."""
    encoder = ParametricFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    assert output.world_scores.requires_grad
    loss = torch.stack(
        [score.raw_nll_or_energy.reshape(-1).mean() for score in output.fit_scores]
    ).sum()
    loss.backward()
    grad = encoder.scale.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all())
    assert float(grad.detach().abs()) > 0.0


def test_world_scores_are_differentiable_and_match_the_detached_readout() -> None:
    model = make_model(dictionary=make_dictionary())
    output = step_bundle(
        single_bundle(), SequenceState(), model, make_history(), ledger=MeasurementLedger()
    )
    assert output.world_scores.shape == (len(output.worlds),)
    assert output.world_scores.requires_grad
    for index, world in enumerate(output.worlds):
        assert output.world_scores[index].detach().item() == pytest.approx(world.score, rel=1e-6)
        assert world.score_kind in {"normalized_density", "empirical_energy", "importance_weight"}
    output.world_scores.sum().backward()
    candidate_grads = [parameter.grad for parameter in model.candidate_encoder.parameters()]
    assert any(grad is not None and bool(torch.isfinite(grad).all()) for grad in candidate_grads)


def _two_camera_bundle(*, cam0_hw=(64, 64), cam1_hw=(48, 64), new_time=True) -> ObservationBundle:
    frame0 = make_frame(values=torch.zeros(*cam0_hw), frame_id="cam0-f0002", camera="cam0")
    frame1 = make_frame(values=torch.ones(*cam1_hw), frame_id="cam1-f0002", camera="cam1")
    return ObservationBundle(
        frames={"cam0": frame0, "cam1": frame1},
        context=EmbodimentContext(),
        episode_ids=("ep0",),
        anchor_ids=("ep0#anchor-f0002",),
        new_time_mask=torch.tensor([new_time]),
    )


def _row_query_for(bundle: ObservationBundle, model: ObservationModel):
    from owac.frameworks.n02_full.model import _build_raw_query, _build_row_query

    stride = int(model.config.query_stride)
    row_query = _build_row_query(
        bundle,
        row=0,
        batch=1,
        episode_id="ep0",
        history=make_history(),
        stride=stride,
        prediction_domain=str(getattr(model.config, "prediction_domain", "normalized")),
        config=model.config,
    )
    return row_query.spec, _build_raw_query(bundle, row_query, row=0, batch=1)


def test_two_camera_bundle_scores_per_camera_without_broadcast() -> None:
    from owac.frameworks.n02_full import sensor as sensor_mod

    model = make_model(
        dictionary=make_dictionary(),
        sensor_model=sensor_mod.SensorModel(units="dn_12bit", density_verified=True),
        sensor_shape={"cam0": (64, 64), "cam1": (48, 64)},
        query_stride=8,
    )
    bundle = _two_camera_bundle()
    ledger = MeasurementLedger()
    output = step_bundle(bundle, SequenceState(), model, make_history(), ledger=ledger)

    assert output.evidence_count == 2  # one measurement per camera
    spec, raw = _row_query_for(bundle, model)
    assert spec.camera_point_counts() == {"cam0": 64, "cam1": 48}
    assert spec.num_points == 112
    assert int(raw.camera_index.shape[-1]) == 112
    assert raw.camera_point_counts() == {"cam0": 64, "cam1": 48}

    cam0 = spec.camera_positions("cam0")
    cam1 = spec.camera_positions("cam1")
    assert torch.equal(cam0, torch.arange(0, 64))
    assert torch.equal(cam1, torch.arange(64, 112))
    coords0 = spec.sensor_coordinates[cam0]
    coords1 = spec.sensor_coordinates[cam1]
    assert float(coords0[:, 0].max()) < 64.0 and float(coords0[:, 1].max()) < 64.0
    assert float(coords1[:, 0].max()) < 48.0 and float(coords1[:, 1].max()) < 64.0

    # full-decision pairing: one score per world, no cross-camera broadcast error
    assert len(output.locked_checks) == len(output.worlds)
    assert len(output.fit_scores) == len(output.worlds)
    assert output.world_scores.shape == (len(output.worlds),)
    assert output.locked_checks[0].raw_nll_or_energy.shape == ()

    # per-camera path: slice both sides, score, and sum into the one objective
    prediction_cam1 = model.predict(
        output.worlds[0], make_history(), spec.select_camera("cam1"), call="fit"
    )
    assert prediction_cam1.mean.shape == (1, 48)
    assert prediction_cam1.masks["camera_index"].tolist() == [[0] * 48]
    cam1_score = model.score(prediction_cam1, raw.select_camera("cam1"))
    cam0_score = model.score(
        model.predict(output.worlds[0], make_history(), spec.select_camera("cam0"), call="fit"),
        raw.select_camera("cam0"),
    )
    combined = cam0_score.raw_nll_or_energy + cam1_score.raw_nll_or_energy
    assert bool(torch.isfinite(combined))
    # a per-camera prediction must not be scored against the whole decision
    with pytest.raises(QueryAlignmentError, match="query points"):
        model.score(prediction_cam1, raw)


# --------------------------------------------------------------------------------------
# fix round 2: units / scale and CUDA device consistency
# --------------------------------------------------------------------------------------

CALIBRATION_12BIT = {"black_level": 64.0, "white_level": 4095.0, "bit_depth": 12}


def test_step_scores_calibrated_frames_in_declared_units() -> None:
    """A calibrated (normalized-internal) step stays finite instead of the 1e8 mismatch."""
    from owac.frameworks.n02_full import sensor as sensor_mod

    model = make_model(
        dictionary=make_dictionary(),
        sensor_model=sensor_mod.SensorModel(units="dn_12bit", density_verified=True),
    )
    bundle = single_bundle(values=torch.full((4, 4), 2000.0), calibration=CALIBRATION_12BIT)
    spec, raw = _row_query_for(bundle, model)
    assert spec.unit_affine is not None
    affine = spec.unit_affine[CAMERA]
    assert affine.scale == pytest.approx(4095.0 - 64.0)
    assert affine.offset == pytest.approx(64.0)
    assert raw.unit_affine is not None

    output = step_bundle(bundle, SequenceState(), model, make_history(), ledger=MeasurementLedger())
    per_point = float(output.locked_checks[0].raw_nll_or_energy.detach()) / spec.num_points
    assert math.isfinite(per_point)
    assert per_point < 1e3  # internal-domain scoring would be ~1e9
    assert bool(torch.isfinite(output.world_scores).all())


def test_prediction_domain_units_declares_no_affine() -> None:
    model = make_model(
        dictionary=make_dictionary(),
        prediction_domain="units",
    )
    bundle = single_bundle(values=torch.full((4, 4), 2000.0), calibration=CALIBRATION_12BIT)
    spec, raw = _row_query_for(bundle, model)
    assert spec.unit_affine is None
    assert raw.unit_affine is None
    output = step_bundle(bundle, SequenceState(), model, make_history(), ledger=MeasurementLedger())
    assert bool(torch.isfinite(output.world_scores).all())


def test_unit_affine_default_config_covers_frames_without_calibration() -> None:
    model = make_model(
        dictionary=make_dictionary(),
        unit_affine_default={"scale": 255.0, "offset": 0.0},
    )
    bundle = single_bundle(values=torch.full((4, 4), 128.0))  # no calibration declared
    spec, _ = _row_query_for(bundle, model)
    assert spec.unit_affine is not None
    assert spec.unit_affine[CAMERA].scale == pytest.approx(255.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_full_measurement_step_runs_on_cuda() -> None:
    from owac.frameworks.n02_full import sensor as sensor_mod

    model = make_model(
        dictionary=make_dictionary(),
        sensor_model=sensor_mod.SensorModel(units="dn_12bit", density_verified=True),
        sensor_shape={"cam0": (4, 4)},
    )
    model.decoder.to("cuda")
    model.candidate_encoder.to("cuda")

    bundle = single_bundle(values=torch.full((4, 4), 2000.0), calibration=CALIBRATION_12BIT)
    for frame in bundle.frames.values():
        frame.values = frame.values.cuda()
        frame.valid_mask = frame.valid_mask.cuda()
        frame.bad_mask = frame.bad_mask.cuda()
        frame.saturated_mask = frame.saturated_mask.cuda()
    bundle.new_time_mask = bundle.new_time_mask.cuda()

    history = make_history()
    history.pooled_features = history.pooled_features.cuda()
    for level in (history.levels or {}).values():
        level.values = level.values.cuda()
        level.valid = level.valid.cuda()
        level.sensor_coordinates = level.sensor_coordinates.cuda()

    output = step_bundle(bundle, SequenceState(), model, history, ledger=MeasurementLedger())
    assert output.world_scores.device.type == "cuda"
    assert output.locked_checks[0].raw_nll_or_energy.device.type == "cuda"
    assert bool(torch.isfinite(output.world_scores).all())

    output.world_scores.sum().backward()
    candidate_grads = [
        parameter.grad
        for parameter in model.candidate_encoder.parameters()
        if parameter.grad is not None
    ]
    decoder_grads = [
        parameter.grad for parameter in model.decoder.parameters() if parameter.grad is not None
    ]
    assert candidate_grads and decoder_grads
    for grad in candidate_grads + decoder_grads:
        assert grad.device.type == "cuda"
        assert bool(torch.isfinite(grad).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_mixed_device_measurement_step_scores_cpu_measurements() -> None:
    """CUDA model over the CPU measurement cache: the micro-batch path must not raise."""
    from owac.frameworks.n02_full import sensor as sensor_mod

    model = make_model(
        dictionary=make_dictionary(),
        sensor_model=sensor_mod.SensorModel(units="dn_12bit", density_verified=True),
        sensor_shape={"cam0": (4, 4)},
    )
    model.decoder.to("cuda")
    model.candidate_encoder.to("cuda")
    bundle = single_bundle(values=torch.full((4, 4), 2000.0), calibration=CALIBRATION_12BIT)
    assert bundle.frames[CAMERA].values.device.type == "cpu"  # cache stays on CPU
    output = step_bundle(bundle, SequenceState(), model, make_history(), ledger=MeasurementLedger())
    assert output.evidence_count == 1
    assert output.world_scores.device.type == "cuda"
    assert bool(torch.isfinite(output.world_scores).all())
    assert bool(torch.isfinite(output.locked_checks[0].raw_nll_or_energy).all())

    output.world_scores.sum().backward()
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.candidate_encoder.parameters()
    )


# --------------------------------------------------------------------------------------
# fix round 6: one batched encode per decision time / per window
# --------------------------------------------------------------------------------------


class ContentFakeEncoder:
    """Features derived from each row's own frame values (row-order independent)."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        self.feature_dim = feature_dim
        self.calls = 0
        self.encoded_rows = 0

    def encode(self, bundle) -> SimpleNamespace:
        from owac.frameworks.n02_full.state import FeatureLevel

        camera_id = next(iter(bundle.frames))
        values = torch.as_tensor(bundle.frames[camera_id].values)
        flat = values.reshape(values.shape[0], -1) if values.dim() >= 3 else values.reshape(1, -1)
        base = flat.mean(dim=1).view(-1, 1, 1, 1).expand(-1, self.feature_dim, 2, 2).clone()
        self.calls += 1
        self.encoded_rows += int(base.shape[0])
        level = FeatureLevel(
            values=base,
            valid=torch.ones(base.shape[0], 2, 2, dtype=torch.bool),
            sensor_coordinates=torch.zeros(base.shape[0], 2, 2, 2),
            support_ids=(),
            level_id="stride8",
        )
        return SimpleNamespace(
            levels={"stride8": level},
            camera_ids=(camera_id,),
            per_camera=lambda: {camera_id: {"stride8": level}},
            input_mode="raw3",
            sensor_shape={},
            support_hash="content",
        )


def test_step_bundle_encodes_new_time_rows_once() -> None:
    encoder = ContentFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    bundle = batch_bundle(
        episodes=("ep-a", "ep-b", "ep-c"),
        frame_ids=("cam0-f0001", "cam0-f0002", "cam0-f0003"),
        new_time=(True, False, True),
    )
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(batch=3, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    assert encoder.calls == 1  # one encode per decision time, not one per row
    assert encoder.encoded_rows == 2  # the repeated row is never encoded
    assert output.cost["fit_context_rows"] == 2.0
    assert output.cost["predictions"] == 2.0 * len(output.worlds)
    assert output.evidence_count == 3  # every row is still registered as a measurement


def test_step_bundle_skips_the_encode_when_no_row_is_new() -> None:
    encoder = ContentFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    bundle = batch_bundle(
        episodes=("ep-a", "ep-b"),
        frame_ids=("cam0-f0001", "cam0-f0002"),
        new_time=(False, False),
    )
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(batch=2, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    assert encoder.calls == 0
    assert output.cost["fit_context_from_current_raw"] == 0.0
    assert output.locked_checks == ()
    assert output.evidence_count == 2


def test_repeat_row_content_does_not_change_new_row_results() -> None:
    encoder = ContentFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    history = make_history(batch=2, last_frame_index=0)
    first = step_bundle(
        batch_bundle(
            episodes=("ep-a", "ep-b"),
            frame_ids=("cam0-f0001", "cam0-f0002"),
            new_time=(True, False),
            rows=[torch.full((4, 4), 10.0), torch.full((4, 4), 1.0)],
        ),
        SequenceState(),
        model,
        history,
        ledger=MeasurementLedger(),
    )
    second = step_bundle(
        batch_bundle(
            episodes=("ep-a", "ep-b"),
            frame_ids=("cam0-f0001", "cam0-f0002"),
            new_time=(True, False),
            rows=[torch.full((4, 4), 10.0), torch.full((4, 4), 999.0)],
        ),
        SequenceState(),
        model,
        history,
        ledger=MeasurementLedger(),
    )
    assert torch.equal(first.world_scores.detach(), second.world_scores.detach())
    for left, right in zip(first.fit_scores, second.fit_scores, strict=True):
        assert torch.equal(left.raw_nll_or_energy, right.raw_nll_or_energy)
    for left, right in zip(first.locked_checks, second.locked_checks, strict=True):
        assert torch.equal(left.raw_nll_or_energy, right.raw_nll_or_energy)


def test_precomputed_window_pyramid_matches_per_step_encoding() -> None:
    """One window encode + row slices reproduces the per-step encode within float noise."""
    from owac.frameworks.n02_full.model import _select_bundle_rows

    encoder = ContentFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    window = batch_bundle(
        episodes=("ep-a", "ep-b", "ep-c", "ep-d"),
        frame_ids=("cam0-f0001", "cam0-f0002", "cam0-f0003", "cam0-f0004"),
        new_time=(True, True, True, True),
        rows=[torch.full((4, 4), float(index + 1)) for index in range(4)],
    )
    steps = [[0, 1], [2, 3]]
    per_step = [
        step_bundle(
            _select_bundle_rows(window, rows),
            SequenceState(),
            model,
            make_history(batch=2, last_frame_index=0),
            ledger=MeasurementLedger(),
        )
        for rows in steps
    ]
    encodes_before = encoder.calls
    pyramid = model.encode(window)
    assert encoder.calls == encodes_before + 1  # the whole window in one forward
    batched = [
        step_bundle(
            _select_bundle_rows(window, rows),
            SequenceState(),
            model,
            make_history(batch=2, last_frame_index=0),
            ledger=MeasurementLedger(),
            fit_pyramid=pyramid,
            fit_rows=rows,
        )
        for rows in steps
    ]
    assert all(item.cost["fit_context_precomputed"] == 1.0 for item in batched)
    assert all(item.cost["fit_context_rows"] == 2.0 for item in batched)
    worst = 0.0
    for reference, candidate in zip(per_step, batched, strict=True):
        assert torch.allclose(
            reference.world_scores.detach(), candidate.world_scores.detach(), rtol=0.0, atol=1e-5
        )
        worst = max(
            worst,
            float((reference.world_scores.detach() - candidate.world_scores.detach()).abs().max()),
        )
        for left, right in zip(reference.fit_scores, candidate.fit_scores, strict=True):
            worst = max(
                worst,
                float((left.raw_nll_or_energy - right.raw_nll_or_energy).abs().max()),
            )
    assert worst < 1e-4, f"window encode vs per-step encode delta {worst}"


# --------------------------------------------------------------------------------------
# round 7: the precomputed window pyramid must carry its multi-scale levels
# --------------------------------------------------------------------------------------


class LevelFakeEncoder:
    """Row-content-derived values/valid/sensor_coordinates; every level field varies."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        self.feature_dim = feature_dim
        self.calls = 0
        self.encoded_rows = 0

    def encode(self, bundle) -> SimpleNamespace:
        from owac.frameworks.n02_full.state import FeatureLevel

        camera_id = next(iter(bundle.frames))
        values = torch.as_tensor(bundle.frames[camera_id].values)
        flat = values.reshape(values.shape[0], -1)
        base = flat.mean(dim=1).view(-1, 1, 1, 1)
        level_values = base.expand(-1, self.feature_dim, 2, 2).clone()
        row = base.reshape(-1)
        valid = row.view(-1, 1, 1).expand(-1, 2, 2).clone() > 0
        coordinates = torch.stack(
            [
                row.view(-1, 1, 1).expand(-1, 2, 2),
                (row * 2.0).view(-1, 1, 1).expand(-1, 2, 2),
            ],
            dim=-1,
        ).clone()
        self.calls += 1
        self.encoded_rows += int(level_values.shape[0])
        level = FeatureLevel(
            values=level_values,
            valid=valid,
            sensor_coordinates=coordinates,
            support_ids=("support",),
            level_id="stride8",
        )
        return SimpleNamespace(
            levels={"stride8": level},
            camera_ids=(camera_id,),
            per_camera=lambda: {camera_id: {"stride8": level}},
            input_mode="raw3",
            sensor_shape={},
            support_hash="level-content",
        )


def _capture_fit_histories(monkeypatch) -> list:
    captured: list = []
    original = ObservationModel.predict

    def spy(self, world, history, query_spec, *, call="prequential_check"):
        if call == "fit":
            captured.append(history)
        return original(self, world, history, query_spec, call=call)

    monkeypatch.setattr(ObservationModel, "predict", spy)
    return captured


def _history_signature(history) -> tuple:
    import hashlib

    def digest(tensor) -> str:
        array = torch.as_tensor(tensor).detach().cpu().numpy()
        return hashlib.sha256(array.tobytes()).hexdigest()

    levels = history.levels or {}
    return (
        digest(history.pooled_features),
        tuple(
            sorted(
                (
                    key,
                    digest(level.values),
                    digest(level.valid),
                    digest(level.sensor_coordinates),
                    tuple(level.support_ids),
                    level.level_id,
                )
                for key, level in levels.items()
            )
        ),
    )


def test_window_pyramid_fit_context_matches_per_step_encoding(monkeypatch) -> None:
    """Window slice vs per-step encode: pooled context and all three level fields agree."""
    from owac.frameworks.n02_full.model import _select_bundle_rows

    encoder = LevelFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    window = batch_bundle(
        episodes=("ep-a", "ep-b", "ep-c", "ep-d"),
        frame_ids=("cam0-f0001", "cam0-f0002", "cam0-f0003", "cam0-f0004"),
        new_time=(True, True, True, True),
        rows=[torch.full((4, 4), float(index + 1)) for index in range(4)],
    )
    steps = [[0, 1], [2, 3]]

    per_step_histories = _capture_fit_histories(monkeypatch)
    for rows in steps:
        step_bundle(
            _select_bundle_rows(window, rows),
            SequenceState(),
            model,
            make_history(batch=2, last_frame_index=0),
            ledger=MeasurementLedger(),
        )
    first = [sorted(map(_history_signature, per_step_histories))]

    del per_step_histories[:]
    pyramid = model.encode(window)
    for rows in steps:
        step_bundle(
            _select_bundle_rows(window, rows),
            SequenceState(),
            model,
            make_history(batch=2, last_frame_index=0),
            ledger=MeasurementLedger(),
            fit_pyramid=pyramid,
            fit_rows=rows,
        )
    windowed = [sorted(map(_history_signature, per_step_histories))]

    assert first == windowed  # bitwise identical pooled features and level triples
    assert first[0], "no fit histories were captured"


def test_repeat_rows_keep_encoding_to_new_rows_with_levels(monkeypatch) -> None:
    encoder = LevelFakeEncoder()
    model = make_model(encoder=encoder, dictionary=make_dictionary())
    bundle = batch_bundle(
        episodes=("ep-a", "ep-b", "ep-c"),
        frame_ids=("cam0-f0001", "cam0-f0002", "cam0-f0003"),
        new_time=(True, False, True),
        rows=[torch.full((4, 4), float(index + 1)) for index in range(3)],
    )
    captured = _capture_fit_histories(monkeypatch)
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(batch=3, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    assert encoder.calls == 1
    assert encoder.encoded_rows == 2  # the repeated row is never encoded
    assert output.cost["fit_context_rows"] == 2.0
    signatures = {_history_signature(history) for history in captured}
    assert signatures
    for signature in signatures:
        pooled_digest, levels = signature
        assert len(levels) == 1  # levels are carried, not dropped
        assert levels[0][1] and levels[0][2] and levels[0][3]  # values/valid/coords present
    pooled_rows = captured[0].pooled_features.shape[0]
    assert pooled_rows == 1  # per-row fit context of the encoded rows only


# --------------------------------------------------------------------------------------
# round 8: the evaluator-facing query builders are the training path's builders
# --------------------------------------------------------------------------------------


def _two_resolution_bundle(*, new_time=(True, False), second_row_value=1.0):
    frame0 = make_frame(
        values=torch.full((2, 4, 4), 10.0), frame_id="cam0-f0001", episode="ep0", camera="cam0"
    )
    frame0.episode_ids = ("ep-a", "ep-b")
    frame0.frame_ids = ("cam0-f0001", "cam0-f0001")
    frame0.new_measurement = torch.tensor(list(new_time))
    second = torch.stack([torch.full((6, 8), 1.0), torch.full((6, 8), float(second_row_value))])
    frame1 = make_frame(values=second, frame_id="cam1-f0001", episode="ep0", camera="cam1")
    frame1.episode_ids = ("ep-a", "ep-b")
    frame1.frame_ids = ("cam1-f0001", "cam1-f0001")
    frame1.new_measurement = torch.tensor(list(new_time))
    return ObservationBundle(
        frames={"cam0": frame0, "cam1": frame1},
        context=EmbodimentContext(),
        episode_ids=("ep-a", "ep-b"),
        anchor_ids=("a0", "a1"),
        new_time_mask=torch.tensor(list(new_time)),
    )


def test_public_query_builders_match_the_training_path(monkeypatch) -> None:
    from owac.frameworks.n02_full import model as model_mod

    captured: dict[str, list] = {"rows": [], "raws": []}
    capturing = {"on": False}
    original_row = model_mod._build_row_query_internal
    original_raw = model_mod._build_raw_query

    def row_spy(bundle, **kwargs):
        value = original_row(bundle, **kwargs)
        if capturing["on"]:
            captured["rows"].append((int(kwargs["row"]), value))
        return value

    def raw_spy(bundle, row_query, *, row, batch):
        value = original_raw(bundle, row_query, row=row, batch=batch)
        if capturing["on"]:
            captured["raws"].append((row, value))
        return value

    monkeypatch.setattr(model_mod, "_build_row_query_internal", row_spy)
    monkeypatch.setattr(model_mod, "_build_raw_query", raw_spy)

    model = make_model(sensor_shape={"cam0": (4, 4), "cam1": (6, 8)}, query_stride=1)
    model.bind_dictionary(make_dictionary())
    bundle = _two_resolution_bundle(new_time=(True, False))
    capturing["on"] = True
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(batch=2, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    capturing["on"] = False
    assert output.evidence_count == 2
    assert [row for row, _ in captured["rows"]] == [0]  # the repeated row is never queried
    assert [row for row, _ in captured["raws"]] == [0]

    # public API, method and module form, reproduce the training-path objects exactly
    for row, training_query in captured["rows"]:
        public_query = model.build_row_query(bundle, row)
        assert public_query.batch == 2
        assert public_query.row == row
        left, right = training_query.spec, public_query.spec
        assert left.keys == right.keys
        assert left.camera_ids == right.camera_ids
        assert left.frame_index == right.frame_index
        assert left.units == right.units
        assert left.unit_affine == right.unit_affine
        assert torch.equal(left.sensor_coordinates, right.sensor_coordinates)
        assert left.camera_point_counts() == {"cam0": 16, "cam1": 48}
        assert right.camera_point_counts() == {"cam0": 16, "cam1": 48}

    for row, training_raw in captured["raws"]:
        public_query = model.build_row_query(bundle, row)
        public_raw = model.build_raw_query(public_query)
        assert public_raw.units == training_raw.units == "dn_12bit"
        assert public_raw.keys == training_raw.keys
        assert torch.equal(public_raw.values, training_raw.values)
        assert torch.equal(public_raw.camera_index, training_raw.camera_index)
        for name in ("valid", "bad", "missing", "saturated"):
            assert torch.equal(public_raw.masks[name], training_raw.masks[name]), name
        # module-level form is the same object shape
        module_raw = model_mod.build_raw_query(model_mod.build_row_query(model, bundle, row))
        assert torch.equal(module_raw.values, public_raw.values)
        assert torch.equal(module_raw.camera_index, public_raw.camera_index)

    # a repeated row still has a well-defined public query (evaluator side)
    repeat_query = model.build_row_query(bundle, 1)
    assert repeat_query.spec.camera_point_counts() == {"cam0": 16, "cam1": 48}
    repeat_raw = model.build_raw_query(repeat_query)
    assert repeat_raw.units == "dn_12bit"
    assert int(repeat_raw.camera_index.shape[-1]) == repeat_raw.num_points
    # the repeated row's content never influenced the encoded/predicted new row
    moved = _two_resolution_bundle(new_time=(True, False), second_row_value=999.0)
    other = step_bundle(
        moved,
        SequenceState(),
        model,
        make_history(batch=2, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    assert torch.equal(output.world_scores.detach(), other.world_scores.detach())


# --------------------------------------------------------------------------------------
# round 9: the per-score hot path must not synchronize the device
# --------------------------------------------------------------------------------------


def _count_sync_calls(monkeypatch) -> dict:
    counts = {"item": 0, "bool": 0}
    original_item = torch.Tensor.item
    original_bool = torch.Tensor.__bool__

    def counted_item(self):
        counts["item"] += 1
        return original_item(self)

    def counted_bool(self):
        counts["bool"] += 1
        return original_bool(self)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)
    return counts


def test_score_hot_path_makes_no_device_synchronization(monkeypatch) -> None:
    """Aligned devices: score() must not call item()/bool(tensor) at all."""
    model = make_model(dictionary=make_dictionary())
    bundle = single_bundle(frame_id="cam0-f0002")
    row_query = model.build_row_query(bundle, 0)
    raw = model.build_raw_query(row_query)
    prediction = model.predict(
        make_dictionary().worlds()[0], make_history(), row_query.spec, call="prequential_check"
    )
    counts = _count_sync_calls(monkeypatch)
    for _ in range(50):
        score = model.score(prediction, raw)
        assert score is not None
    assert counts == {"item": 0, "bool": 0}


def test_align_query_device_returns_the_same_object_when_aligned() -> None:
    model = make_model(dictionary=make_dictionary())
    bundle = single_bundle(frame_id="cam0-f0002")
    row_query = model.build_row_query(bundle, 0)
    raw = model.build_raw_query(row_query)
    prediction = model.predict(
        make_dictionary().worlds()[0], make_history(), row_query.spec, call="prequential_check"
    )
    aligned = model._align_query_device(prediction, raw)
    assert aligned is raw


def _timed_pair(first, second, repeats: int = 3) -> tuple[float, float]:
    """Interleaved medians of two callables (load-robust ratio, single-threaded torch)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    first_times: list[float] = []
    second_times: list[float] = []
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            first()
            first_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            second()
            second_times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(previous)
    first_times.sort()
    second_times.sort()
    return first_times[repeats // 2], second_times[repeats // 2]


def test_score_hot_path_beats_revalidating_every_call() -> None:
    """A per-call RawQuery rebuild (the old behaviour) is measurably slower."""
    from owac.frameworks.n02_full.factors import RawQuery as _RawQuery

    model = make_model(dictionary=make_dictionary())
    bundle = single_bundle(frame_id="cam0-f0002")
    row_query = model.build_row_query(bundle, 0)
    raw = model.build_raw_query(row_query)
    prediction = model.predict(
        make_dictionary().worlds()[0], make_history(), row_query.spec, call="prequential_check"
    )
    repeats = 150

    def hot():
        for _ in range(repeats):
            model.score(prediction, raw)

    def legacy():
        for _ in range(repeats):
            rebuilt = _RawQuery(
                values=raw.values,
                masks=raw.masks,
                units=raw.units,
                keys=raw.keys,
                camera_index=raw.camera_index,
                unit_affine=raw.unit_affine,
            )
            model.score(prediction, rebuilt)

    hot_time, legacy_time = _timed_pair(hot, legacy)
    assert hot_time < 0.9 * legacy_time, (
        f"hot path {hot_time:.4f}s vs revalidating path {legacy_time:.4f}s"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_score_hot_path_makes_no_synchronization_on_cuda(monkeypatch) -> None:
    """On the real device the aligned score path must not synchronize either."""
    model = make_model(dictionary=make_dictionary(), sensor_shape={"cam0": (4, 4)})
    model.decoder.to("cuda")
    model.candidate_encoder.to("cuda")
    bundle = single_bundle(values=torch.full((4, 4), 2000.0), calibration=CALIBRATION_12BIT)
    for frame in bundle.frames.values():
        frame.values = frame.values.cuda()
        frame.valid_mask = frame.valid_mask.cuda()
        frame.bad_mask = frame.bad_mask.cuda()
        frame.saturated_mask = frame.saturated_mask.cuda()
    bundle.new_time_mask = bundle.new_time_mask.cuda()
    history = make_history()
    history.pooled_features = history.pooled_features.cuda()
    for level in (history.levels or {}).values():
        level.values = level.values.cuda()
        level.valid = level.valid.cuda()
        level.sensor_coordinates = level.sensor_coordinates.cuda()
    row_query = model.build_row_query(bundle, 0)
    raw = model.build_raw_query(row_query)
    prediction = model.predict(
        make_dictionary().worlds()[0], history, row_query.spec, call="prequential_check"
    )
    assert prediction.mean.device.type == "cuda"
    counts = _count_sync_calls(monkeypatch)
    for _ in range(20):
        assert model.score(prediction, raw) is not None
    assert counts == {"item": 0, "bool": 0}


def test_step_bundle_synchronizes_only_at_the_step_boundary(monkeypatch) -> None:
    """One step: no per-row/per-candidate item()/bool() calls (only the step readouts)."""
    model = make_model(dictionary=make_dictionary())
    bundle = batch_bundle(
        episodes=("ep-a", "ep-b"),
        frame_ids=("cam0-f0002", "cam0-f0002"),
        new_time=(True, True),
    )
    counts = _count_sync_calls(monkeypatch)
    output = step_bundle(
        bundle,
        SequenceState(),
        model,
        make_history(batch=2, last_frame_index=0),
        ledger=MeasurementLedger(),
    )
    assert counts["item"] == 0  # no index max()/item() validation on constructed queries
    assert counts["bool"] <= 4, counts  # only the per-step flags/reductions
    assert output.world_scores.requires_grad

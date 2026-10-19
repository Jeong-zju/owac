"""P1 factors tests: fit/prequential causality, candidate dictionary, conditional decoder.

Every assertion here is a semantic contract from the freeze; nothing is an import
smoke test. The sensor model and the raw encoder are lightweight same-signature
fakes because ``sensor.py`` (T2) and ``observation.py`` (T1) are developed in the
same batch; the integration check against the real modules is a separate run.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.factors import (
    CANDIDATE_CATEGORIES,
    AnalyticOracle,
    CandidateDictionary,
    CandidateGeometry,
    CandidateSpec,
    CausalLeakError,
    HistoryContext,
    ObservationModel,
    QueryAlignmentError,
    QuerySpec,
    RawQuery,
    UnitAffine,
    _indices_match,
    build_p1_task,
    pool_feature_pyramid,
)
from owac.frameworks.n02_full.model import step_bundle
from owac.frameworks.n02_full.state import (
    EntityState,
    FeatureLevel,
    InferenceInput,
    MeasurementKey,
    MeasurementLedger,
    ObservationBundle,
    SequenceState,
    TrainingTargets,
    WorldHypothesis,
)

FEATURE_DIM = 8
WORLD_SIZE = 2
CANDIDATE_EMBEDDING_DIM = 8
DECODER_GRID = 4
DECODER_CHANNELS = 4
DECODER_HIDDEN = 8
SENSOR_SHAPE = {"cam0": (4, 4)}


# --------------------------------------------------------------------------------------
# same-signature fakes for the T1/T2 modules that land in the same batch
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
    """Gaussian proper score, same shape contract as the real sensor model."""

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
    """Returns one declared stride8 level; records the calls it received."""

    def __init__(self, feature_dim: int = FEATURE_DIM, value: float = 2.0, support_ids=()) -> None:
        self.feature_dim = feature_dim
        self.value = float(value)
        self.support_ids = tuple(support_ids)
        self.calls = 0

    def encode(self, bundle) -> SimpleNamespace:
        self.calls += 1
        batch = len(bundle.episode_ids) or 1
        height, width = 2, 3
        level = FeatureLevel(
            values=torch.full((batch, self.feature_dim, height, width), self.value),
            valid=torch.ones((batch, height, width), dtype=torch.bool),
            sensor_coordinates=torch.zeros((batch, height, width, 2)),
            support_ids=self.support_ids,
            level_id="stride8",
        )
        cameras = tuple(bundle.frames)
        return SimpleNamespace(
            levels={"stride8": level, "stride16": level},
            camera_ids=cameras,
            per_camera=lambda: {
                camera: {"stride8": level, "stride16": level} for camera in cameras
            },
            input_mode="raw3",
            sensor_shape={},
            support_hash="fake",
        )


def make_config(**overrides) -> SimpleNamespace:
    values = {
        "feature_dim": FEATURE_DIM,
        "candidate_world_size": WORLD_SIZE,
        "candidate_embedding_dim": CANDIDATE_EMBEDDING_DIM,
        "decoder_grid": DECODER_GRID,
        "decoder_channels": DECODER_CHANNELS,
        "decoder_hidden": DECODER_HIDDEN,
        "sensor_shape": dict(SENSOR_SHAPE),
        "structure_version": "p1-test-structure",
        "initial_prior_version": "prior-v1",
        "query_stride": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_model(*, encoder=None, sensor_model=None, **config_overrides) -> ObservationModel:
    torch.manual_seed(1234)
    return ObservationModel(
        encoder if encoder is not None else FakeEncoder(),
        sensor_model if sensor_model is not None else FakeSensorModel(),
        make_config(**config_overrides),
        version="p1-test-model",
    )


# --------------------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------------------

TRUTH = torch.tensor([[1.0, 1.0], [3.0, 2.0]], dtype=torch.float32)


def geometry(positions) -> CandidateGeometry:
    return CandidateGeometry(world_px=torch.as_tensor(positions, dtype=torch.float32))


def wrong_specs(*, offsets=((1.0, 0.0), (0.0, -2.0)), base=None) -> list[CandidateSpec]:
    positions = TRUTH if base is None else torch.as_tensor(base, dtype=torch.float32)
    specs = []
    for index, offset in enumerate(offsets):
        shifted = positions.clone()
        shifted[index % positions.shape[0]] += torch.as_tensor(offset, dtype=torch.float32)
        specs.append(
            CandidateSpec(
                candidate_id=f"wrong-{index:02d}",
                category="wrong",
                geometry=geometry(shifted),
                nuisance={"log_exposure": 0.2 * (index + 1)},
            )
        )
    return specs


def make_dictionary(
    *,
    all_wrong: bool = False,
    correct_index: int = 0,
    oracle=None,
    truth=None,
    offsets=((1.0, 0.0), (0.0, -2.0)),
) -> CandidateDictionary:
    positions = TRUTH if truth is None else torch.as_tensor(truth, dtype=torch.float32)
    wrongs = wrong_specs(offsets=offsets, base=positions)
    if all_wrong:
        return CandidateDictionary(specs=tuple(wrongs), all_wrong=True, oracle=oracle)
    correct = CandidateSpec(
        candidate_id="correct", category="correct", geometry=geometry(positions.clone())
    )
    specs = tuple(wrongs[:correct_index]) + (correct,) + tuple(wrongs[correct_index:])
    return CandidateDictionary(specs=specs, oracle=oracle)


def make_history(
    *,
    batch: int = 1,
    last_frame_index: int | None = 0,
    feature: float = 0.5,
    pooled: torch.Tensor | None = None,
    support_ids=(),
    eta=None,
    eta_fit_keys=(),
) -> HistoryContext:
    features = torch.full((batch, FEATURE_DIM), feature) if pooled is None else pooled
    levels = {
        "stride8": FeatureLevel(
            values=features.unsqueeze(-1).unsqueeze(-1),
            valid=torch.ones((batch, 1, 1), dtype=torch.bool),
            sensor_coordinates=torch.zeros((batch, 1, 1, 2)),
            support_ids=tuple(support_ids),
            level_id="stride8",
        )
    }
    return HistoryContext(
        pooled_features=features,
        levels=levels,
        episode_ids=("ep0",) * batch,
        anchor_ids=("ep0#anchor",) * batch,
        last_frame_index=last_frame_index,
        eta=eta,
        eta_fit_keys=eta_fit_keys,
    )


def make_query(
    *,
    frame_index: int = 1,
    camera: str = "cam0",
    source_frame_id: str = "cam0-f0001",
    units: str = "dn_12bit",
    points=((-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)),
) -> QuerySpec:
    coordinates = torch.tensor(points, dtype=torch.float32)
    key = MeasurementKey(
        episode_id="ep0",
        camera_id=camera,
        source_frame_id=source_frame_id,
        acquisition_id=source_frame_id,
    )
    return QuerySpec(
        keys=(key,),
        sensor_coordinates=coordinates,
        camera_ids=(camera,) * int(coordinates.shape[0]),
        frame_index=frame_index,
        units=units,
    )


# --------------------------------------------------------------------------------------
# causality: prequential vs fit
# --------------------------------------------------------------------------------------


def test_prequential_prediction_ignores_current_query_values() -> None:
    """The prediction interface has no path from y_t; swapping y_t changes scoring only."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    history = make_history()
    spec = make_query()

    first = model.predict(world, history, spec, call="prequential_check")
    raw_low = RawQuery(values=torch.zeros(spec.num_points), units=spec.units, keys=spec.keys)
    raw_high = RawQuery(
        values=torch.full((spec.num_points,), 500.0), units=spec.units, keys=spec.keys
    )
    low = model.score(first, raw_low)
    high = model.score(first, raw_high)
    assert not torch.allclose(low.raw_nll_or_energy, high.raw_nll_or_energy)

    second = model.predict(world, history, spec, call="prequential_check")
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.variance, second.variance)

    parameters = set(inspect.signature(ObservationModel.predict).parameters) - {"self"}
    assert parameters == {"world", "history", "query_spec", "call"}


def test_prequential_refuses_history_that_reaches_the_query_frame() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=3)
    with pytest.raises(CausalLeakError):
        model.predict(world, make_history(last_frame_index=3), spec, call="prequential_check")
    prediction = model.predict(world, make_history(last_frame_index=3), spec, call="fit")
    assert prediction.mean.shape == (1, spec.num_points)


def test_prequential_refuses_support_that_belongs_to_the_query() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=5, source_frame_id="cam0-f0004")
    history = make_history(last_frame_index=4, support_ids=("cam0-f0004",))
    with pytest.raises(CausalLeakError):
        model.predict(world, history, spec, call="prequential_check")


def test_fit_may_use_the_target_set_while_prequential_may_not() -> None:
    """Behaviourally decidable: the same history is accepted by fit and refused by prequential."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=1)
    history_only = model.predict(
        world, make_history(last_frame_index=0, feature=0.25), spec, call="fit"
    )
    target_set = model.predict(
        world, make_history(last_frame_index=1, feature=3.0), spec, call="fit"
    )
    assert not torch.allclose(history_only.mean, target_set.mean)
    with pytest.raises(CausalLeakError):
        model.predict(
            world, make_history(last_frame_index=1, feature=3.0), spec, call="prequential_check"
        )


def test_first_frame_is_calibration_only() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=0)
    prediction = model.predict(
        world, make_history(last_frame_index=None), spec, call="prequential_check"
    )
    assert bool(prediction.masks["calibration_only"].all())
    fit = model.predict(world, make_history(last_frame_index=None), spec, call="fit")
    assert not bool(fit.masks["calibration_only"].any())


def test_variance_is_positive_and_aligned_with_query_points() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query()
    prediction = model.predict(world, make_history(), spec, call="prequential_check")
    assert prediction.mean.shape == (1, spec.num_points)
    assert torch.isfinite(prediction.mean).all()
    assert bool((prediction.variance > 0).all())
    assert prediction.units == spec.units


def test_score_refuses_different_units() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(units="dn_12bit")
    prediction = model.predict(world, make_history(), spec, call="prequential_check")
    other = RawQuery(values=torch.zeros(spec.num_points), units="dn_16bit")
    with pytest.raises(ValueError):
        model.score(prediction, other)


def test_unknown_call_is_refused() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    with pytest.raises(ValueError):
        model.predict(world, make_history(), make_query(), call="bayes")


# --------------------------------------------------------------------------------------
# candidate dictionary
# --------------------------------------------------------------------------------------


def test_dictionary_carries_correct_wrong_and_all_wrong_classes() -> None:
    dictionary = make_dictionary()
    assert dictionary.categories() == {"correct": 1, "wrong": 2}
    assert all(spec.category in CANDIDATE_CATEGORIES for spec in dictionary.specs)

    all_wrong = make_dictionary(all_wrong=True)
    assert all_wrong.categories() == {"correct": 0, "wrong": 2}
    assert all_wrong.label_index(TRUTH) is None

    with pytest.raises(ValueError):
        CandidateDictionary(specs=(CandidateSpec("only", "wrong", geometry(TRUTH)),))
    with pytest.raises(ValueError):
        CandidateDictionary(
            specs=tuple(wrong_specs()) + (CandidateSpec("c", "correct", geometry(TRUTH)),),
            all_wrong=True,
        )


def test_label_index_is_constructive_not_nearest_neighbour() -> None:
    """The label is fixed at construction; the nearest neighbour is only a cross-check."""
    near = TRUTH + 0.25
    dictionary = CandidateDictionary(
        specs=(
            CandidateSpec("far", "wrong", geometry(TRUTH + 5.0)),
            CandidateSpec("declared", "correct", geometry(near)),
            CandidateSpec("exact", "wrong", geometry(TRUTH)),
        )
    )
    assert dictionary.correct_index == 1
    assert dictionary.label_index(TRUTH) is None  # exact cross-check failed -> unavailable
    report = dictionary.cross_check(TRUTH)
    assert report["nearest_index"] == 2  # a nearest-neighbour label would have said 2
    assert report["exact_match"] is False

    matching = CandidateDictionary(
        specs=(
            CandidateSpec("far", "wrong", geometry(TRUTH + 5.0)),
            CandidateSpec("declared", "correct", geometry(TRUTH.clone())),
            CandidateSpec("other", "wrong", geometry(TRUTH + 1.0)),
        )
    )
    assert matching.label_index(TRUTH) == 1
    assert matching.label_index(None) == 1  # constructive mapping without a cross-check
    assert matching.cross_check(TRUTH)["exact_match"] is True


def test_candidate_capacity_is_equal_and_bound_to_the_model() -> None:
    model = make_model()
    dictionary = make_dictionary()
    model.bind_dictionary(dictionary)
    assert dictionary.capacity() == model.candidate_dim
    assert len({spec.geometry.world_size for spec in dictionary.specs}) == 1
    with pytest.raises(ValueError):
        model.bind_dictionary(make_dictionary(truth=torch.tensor([[1.0, 1.0]])))


def test_candidates_share_one_decoder_and_have_no_per_candidate_parameters() -> None:
    model = make_model()
    dictionary = make_dictionary()
    model.bind_dictionary(dictionary)
    worlds = dictionary.worlds()
    history = make_history()
    spec = make_query()
    first = model.predict(worlds[0], history, spec, call="prequential_check")
    second = model.predict(worlds[0], history, spec, call="prequential_check")
    assert torch.equal(first.mean, second.mean)

    parameter_names = " ".join(name for name, _ in model.decoder.named_parameters())
    for candidate in dictionary.specs:
        assert candidate.candidate_id not in parameter_names

    # equal geometry -> bit-identical prediction: no hidden per-candidate path
    twin = CandidateSpec("wrong-twin", "wrong", geometry(dictionary.specs[0].geometry.world_px))
    twin_world = twin.as_world()
    original = dictionary.specs[0].as_world()
    assert torch.equal(
        model.predict(twin_world, history, spec, call="prequential_check").mean,
        model.predict(original, history, spec, call="prequential_check").mean,
    )


def test_candidate_cannot_smuggle_current_query_or_truth() -> None:
    with pytest.raises(ValueError, match="could copy the current query"):
        CandidateSpec(
            "wrong",
            "wrong",
            geometry(TRUTH),
            nuisance={"raw_values": torch.zeros(3)},
        )
    with pytest.raises(ValueError, match="unknown nuisance fields"):
        CandidateSpec("wrong", "wrong", geometry(TRUTH), nuisance={"exposure": 1.0})

    model = make_model()
    history = make_history()
    spec = make_query()
    entity = EntityState(
        entity_ids=("w",), geometry={"world_px": TRUTH.clone()}, geometry_frame="view_2d"
    )
    bad_world = WorldHypothesis(
        world_id="w",
        entities=(entity,),
        nuisance={"query_values": torch.zeros(2)},
    )
    with pytest.raises(ValueError, match="could copy the current query"):
        model.predict(bad_world, history, spec, call="prequential_check")


def test_world_noise_scale_and_depth_enter_the_candidate_embedding() -> None:
    model = make_model()
    history = make_history()
    spec = make_query()
    base = make_dictionary().specs[0].as_world()
    scaled = CandidateSpec(
        "wrong-scaled",
        "wrong",
        CandidateGeometry(world_px=TRUTH.clone(), scale_px=torch.tensor([1.5, 0.5])),
    ).as_world()
    assert not torch.allclose(
        model.predict(base, history, spec, call="prequential_check").mean,
        model.predict(scaled, history, spec, call="prequential_check").mean,
    )


# --------------------------------------------------------------------------------------
# interference eta
# --------------------------------------------------------------------------------------


def test_history_eta_scales_mean_and_variance_and_is_tagged() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=3)
    plain = model.predict(world, make_history(last_frame_index=2), spec, call="prequential_check")
    eta = FakeInterferenceEstimate(
        log_exposure=torch.tensor([math.log(2.0)]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([1.0]),
        source="history",
    )
    exposed = model.predict(
        world, make_history(last_frame_index=2, eta=eta), spec, call="prequential_check"
    )
    assert torch.allclose(exposed.mean, plain.mean * 2.0, rtol=1e-5, atol=1e-6)
    assert not bool(exposed.masks["oracle_eta"].any())
    assert not bool(exposed.masks["empirical_check"].any())

    noisy = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([2.0]),
        source="history",
    )
    scaled = model.predict(
        world, make_history(last_frame_index=2, eta=noisy), spec, call="prequential_check"
    )
    assert torch.allclose(scaled.variance, plain.variance * 4.0, rtol=1e-5)

    oracle = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([1.0]),
        source="oracle",
    )
    tagged = model.predict(
        world, make_history(last_frame_index=2, eta=oracle), spec, call="prequential_check"
    )
    assert bool(tagged.masks["oracle_eta"].all())


def test_eta_fitted_on_the_query_marks_the_check_empirical() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=3)
    eta = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.zeros(1, 2),
        noise_scale=torch.tensor([1.0]),
        source="history",
    )
    prediction = model.predict(
        world,
        make_history(last_frame_index=2, eta=eta, eta_fit_keys=spec.keys),
        spec,
        call="prequential_check",
    )
    assert bool(prediction.masks["empirical_check"].all())


def test_eta_translation_shifts_the_sampling_coordinates() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query(frame_index=3)
    plain = model.predict(world, make_history(last_frame_index=2), spec, call="prequential_check")
    shifted = FakeInterferenceEstimate(
        log_exposure=torch.tensor([0.0]),
        camera_translation=torch.tensor([[1.0, -1.0]]),
        noise_scale=torch.tensor([1.0]),
        source="history",
    )
    moved = model.predict(
        world, make_history(last_frame_index=2, eta=shifted), spec, call="prequential_check"
    )
    assert not torch.allclose(plain.mean, moved.mean)


# --------------------------------------------------------------------------------------
# analytic oracle
# --------------------------------------------------------------------------------------


def test_oracle_is_never_a_dictionary_member_and_needs_explicit_attachment() -> None:
    dictionary = make_dictionary()
    with pytest.raises(LookupError):
        dictionary.oracle_world()

    def mean_variance(history, query_spec):
        return torch.zeros(1, query_spec.num_points), torch.ones(1, query_spec.num_points)

    oracle = AnalyticOracle(
        mean_variance, prediction_type=FakeRawPrediction, sensor_model=FakeSensorModel()
    )
    with_oracle = make_dictionary(oracle=oracle)
    world = with_oracle.oracle_world()
    assert world.world_id not in {item.world_id for item in with_oracle.worlds()}

    spec = make_query()
    prediction = oracle.predict(world, make_history(), spec, call="prequential_check")
    assert bool(prediction.masks["oracle_eta"].all())
    assert prediction.units == spec.units
    score = oracle.score(
        prediction, RawQuery(values=torch.zeros(spec.num_points), units=spec.units)
    )
    assert score.raw_nll_or_energy.shape == (1, spec.num_points)


# --------------------------------------------------------------------------------------
# encoder delegation and feature pooling
# --------------------------------------------------------------------------------------


def test_encode_delegates_to_the_raw_encoder() -> None:
    encoder = FakeEncoder()
    model = make_model(encoder=encoder)
    bundle = SimpleNamespace(episode_ids=("ep0",), frames={"cam0": object()})
    pyramid = model.encode(bundle)
    assert encoder.calls == 1
    assert set(pyramid.levels) == {"stride8", "stride16"}


def test_pool_feature_pyramid_ignores_invalid_positions() -> None:
    level = FeatureLevel(
        values=torch.tensor([[[[1.0, 1.0], [1000.0, 1.0]]]]),
        valid=torch.tensor([[[True, True], [False, True]]]),
        sensor_coordinates=torch.zeros(1, 2, 2, 2),
        support_ids=(),
        level_id="stride8",
    )
    pooled = pool_feature_pyramid(SimpleNamespace(levels={"stride8": level}))
    assert pooled.shape == (1, 1)
    assert float(pooled[0, 0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# controlled-task assembly
# --------------------------------------------------------------------------------------


def _task_source(*, frames=None, truth=TRUTH, targets=None):
    frame = MeasurementFrame(
        values=torch.arange(16, dtype=torch.float32).reshape(4, 4) if frames is None else frames,
        valid_mask=torch.ones(4, 4, dtype=torch.bool),
        bad_mask=torch.zeros(4, 4, dtype=torch.bool),
        saturated_mask=torch.zeros(4, 4, dtype=torch.bool),
        episode_ids=("ep0",),
        frame_ids=("cam0-f0001",),
        camera_id="cam0",
        new_measurement=True,
        metadata={"unit": "dn_12bit"},
    )
    bundle = SimpleNamespace(frames={"cam0": frame}, new_time_mask=torch.tensor([True]))
    inference = InferenceInput(bundle=bundle)
    truth_record = SimpleNamespace(geometry={"world_px": truth.clone()})
    return SimpleNamespace(
        inference=[inference],
        truth=[truth_record],
        targets=[TrainingTargets()] if targets is None else targets,
    )


def test_build_p1_task_labels_the_correct_candidate_constructively() -> None:
    task = build_p1_task(_task_source(), correct_index=2)
    assert task.candidate_index == 2
    assert task.dictionary.correct_index == 2
    assert task.dictionary.categories() == {"correct": 1, "wrong": 3}
    geometry_target = task.targets.optional_geometry
    assert geometry_target["candidate_index"] == 2
    assert geometry_target["geometry_frame"] == "view_2d"
    assert torch.equal(geometry_target["world_px"], TRUTH)
    assert task.units == "dn_12bit"
    for world in task.worlds:
        assert set(world.entities[0].geometry) == {"world_px", "scale_px", "depth"}
        assert world.entities[0].geometry_frame == "view_2d"


def test_build_p1_task_all_wrong_has_no_fabricated_label() -> None:
    task = build_p1_task(_task_source(), all_wrong=True)
    assert task.candidate_index is None
    assert task.dictionary.all_wrong is True
    assert "candidate_index" not in task.targets.optional_geometry
    assert torch.equal(task.targets.optional_geometry["world_px"], TRUTH)
    assert task.dictionary.label_index(task.truth) is None


# --------------------------------------------------------------------------------------
# integration with the delivered T1 module
# --------------------------------------------------------------------------------------


def _real_level(value: float, shape, level_id: str) -> FeatureLevel:
    batch, channels, height, width = shape
    return FeatureLevel(
        values=torch.full((batch, channels, height, width), float(value)),
        valid=torch.ones((batch, height, width), dtype=torch.bool),
        sensor_coordinates=torch.zeros((batch, height, width, 2)),
        support_ids=(),
        level_id=level_id,
    )


def test_pool_feature_pyramid_uses_the_declared_per_camera_view() -> None:
    """T1's multi-camera keys are consumed through ``per_camera()``, never parsed by hand."""
    from owac.frameworks.n02_full.observation import FeaturePyramid as RealFeaturePyramid

    pyramid = RealFeaturePyramid(
        levels={
            "cam_a::stride8": _real_level(1.0, (1, 2, 2, 2), "stride8"),
            "cam_a::stride16": _real_level(1.0, (1, 2, 1, 1), "stride16"),
            "cam_b::stride8": _real_level(7.0, (1, 2, 2, 2), "stride8"),
            "cam_b::stride16": _real_level(7.0, (1, 2, 1, 1), "stride16"),
        },
        camera_ids=("cam_a", "cam_b"),
        input_mode="raw3",
        sensor_shape={"cam_a": (64, 64), "cam_b": (32, 32)},
        support_hash="fixture",
    )
    assert float(pool_feature_pyramid(pyramid)[0, 0]) == pytest.approx(1.0)
    assert float(pool_feature_pyramid(pyramid, camera_id="cam_b")[0, 0]) == pytest.approx(7.0)
    assert pool_feature_pyramid(pyramid, camera_id="cam_b", level_id="stride16").shape == (1, 2)

    single = RealFeaturePyramid(
        levels={
            "stride8": _real_level(3.0, (1, 2, 2, 2), "stride8"),
            "stride16": _real_level(3.0, (1, 2, 1, 1), "stride16"),
        },
        camera_ids=("only",),
        input_mode="raw3",
        sensor_shape={"only": (32, 32)},
        support_hash="fixture",
    )
    assert float(pool_feature_pyramid(single)[0, 0]) == pytest.approx(3.0)


def test_build_p1_task_on_a_real_generator_sequence() -> None:
    """End-to-end assembly against the P0 generator: inference, targets, truth, dictionary."""
    from owac.frameworks.n02_full import synthetic as synth

    sensor = synth.SensorSpec(
        camera_id="cam0",
        height=16,
        width=16,
        input_mode="rawvla_rgb3",
        black_level=0.0,
        white_level=4095.0,
        bit_depth=16,
        read_noise_e=0.0,
    )
    shape = synth.ShapeSpec(
        shape_id="obj",
        kind="disc",
        scale_px=3.0,
        position_px=(8.0, 8.0),
        velocity_px_per_step=(0.5, 0.0),
        colour_response=(0.8, 0.6, 0.4),
    )
    scene = synth.SceneSpec(
        scene_id="p1-scene",
        seed=7,
        partition="dev",
        shapes=(shape,),
        background_level=0.2,
        camera_motion_px=(0.0, 0.0),
    )
    measurement = synth.MeasurementModel(
        kind="raw3", shot_noise_scale=1000.0, read_noise_e=1.0, response=(0.8, 0.6, 0.4)
    )
    schedule = synth.EventSchedule(horizon=4, seed=11)
    sequence = synth.generate_sequence(scene, (sensor,), measurement, schedule, frames=4, seed=11)

    task = build_p1_task(sequence, frame_index=2, correct_index=1)
    assert task.candidate_index == 1
    assert task.targets.optional_geometry["candidate_index"] == 1
    assert task.candidate_index == task.dictionary.label_index(task.truth)
    assert task.units == "pseudo_raw_dn_float32"
    assert task.inference.bundle.frames["cam0"].metadata["unit"] == task.units
    assert torch.equal(
        task.targets.optional_geometry["world_px"],
        torch.as_tensor(task.truth.geometry["world_px"], dtype=torch.float32),
    )
    assert task.dictionary.capacity() == 4 * task.dictionary.world_size() + 4


# --------------------------------------------------------------------------------------
# fix round 1: gradient path and per-camera pairing
# --------------------------------------------------------------------------------------


def test_pooled_features_keep_the_graph_and_the_values() -> None:
    """A learnable pooled context is re-typed, never detached; values are unchanged."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query()
    features = torch.full((1, FEATURE_DIM), 0.5, requires_grad=True)
    with_grad = model.predict(world, make_history(pooled=features), spec, call="fit")
    assert with_grad.mean.requires_grad
    with_grad.mean.sum().backward()
    assert features.grad is not None
    assert bool(torch.isfinite(features.grad).all())

    detached = model.predict(world, make_history(pooled=features.detach()), spec, call="fit")
    assert torch.equal(with_grad.mean.detach(), detached.mean)
    assert torch.equal(with_grad.variance.detach(), detached.variance)


def _two_camera_spec() -> QuerySpec:
    keys = (
        MeasurementKey("ep0", "cam0", "cam0-f0002", "cam0-f0002"),
        MeasurementKey("ep0", "cam1", "cam1-f0002", "cam1-f0002"),
    )
    coordinates = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [0.5, 0.5]], dtype=torch.float32
    )
    return QuerySpec(
        keys=keys,
        sensor_coordinates=coordinates,
        camera_ids=("cam0", "cam0", "cam0", "cam1"),
        frame_index=2,
        units="dn_12bit",
    )


def test_query_spec_and_raw_query_carry_the_per_point_camera_index() -> None:
    spec = _two_camera_spec()
    assert spec.camera_point_counts() == {"cam0": 3, "cam1": 1}
    assert spec.camera_index().tolist() == [0, 0, 0, 1]
    assert spec.camera_positions("cam1").tolist() == [3]
    cam1 = spec.select_camera("cam1")
    assert cam1.num_points == 1
    assert cam1.camera_ids == ("cam1",)
    assert cam1.keys[0].camera_id == "cam1"
    assert cam1.camera_index().tolist() == [0]
    assert cam1.sensor_coordinates.tolist() == [[0.5, 0.5]]

    raw = RawQuery(
        values=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        masks={
            "valid": torch.ones(1, 4, dtype=torch.bool),
            "bad": torch.zeros(1, 4, dtype=torch.bool),
            "missing": torch.zeros(1, 4, dtype=torch.bool),
            "saturated": torch.zeros(1, 4, dtype=torch.bool),
        },
        units="dn_12bit",
        keys=spec.keys,
        camera_index=spec.camera_index(),
    )
    assert raw.camera_point_counts() == {"cam0": 3, "cam1": 1}
    sliced = raw.select_camera("cam1")
    assert sliced.num_points == 1
    assert torch.equal(sliced.values, torch.tensor([[3.0]]))
    assert sliced.camera_point_counts() == {"cam1": 1}
    assert sliced.keys[0].camera_id == "cam1"


def test_score_refuses_cross_camera_pairing_instead_of_broadcasting() -> None:
    model = make_model(sensor_shape={"cam0": (4, 4), "cam1": (4, 4)})
    world = make_dictionary().worlds()[0]
    history = make_history()
    spec = _two_camera_spec()
    raw = RawQuery(
        values=torch.zeros(1, spec.num_points),
        masks={"valid": torch.ones(1, spec.num_points, dtype=torch.bool)},
        units=spec.units,
        keys=spec.keys,
        camera_index=spec.camera_index(),
    )
    full = model.predict(world, history, spec, call="fit")
    model.score(full, raw)  # aligned multi-camera pairing is fine

    single = model.predict(world, history, spec.select_camera("cam0"), call="fit")
    assert single.mean.shape == (1, 3)
    with pytest.raises(QueryAlignmentError):
        model.score(single, raw)
    # one camera at a time is the supported path and sums into the same objective
    per_camera = model.score(single, raw.select_camera("cam0"))
    assert int(per_camera.raw_nll_or_energy.shape[-1]) == 3

    # equal point counts but a different per-point camera pattern is still refused
    declared = QuerySpec(
        keys=spec.keys,
        sensor_coordinates=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        camera_ids=("cam0", "cam1"),
        frame_index=2,
        units="dn_12bit",
    )
    crossed = QuerySpec(
        keys=spec.keys,
        sensor_coordinates=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        camera_ids=("cam1", "cam0"),
        frame_index=2,
        units="dn_12bit",
    )
    raw_declared = RawQuery(
        values=torch.zeros(1, 2),
        masks={"valid": torch.ones(1, 2, dtype=torch.bool)},
        units="dn_12bit",
        keys=declared.keys,
        camera_index=declared.camera_index(),
    )
    crossed_prediction = model.predict(world, history, crossed, call="fit")
    assert crossed_prediction.masks["camera_index"].tolist() == [[1, 0]]
    with pytest.raises(QueryAlignmentError, match="camera"):
        model.score(crossed_prediction, raw_declared)


def test_prediction_is_never_broadcast_across_query_points() -> None:
    model = make_model()
    world = make_dictionary().worlds()[0]
    single = model.predict(world, make_history(), make_query(points=((0.5, 0.5),)), call="fit")
    many = make_query(points=((0.5, 0.5), (1.5, 0.5), (2.5, 0.5)))
    raw_many = RawQuery(values=torch.zeros(1, 3), units="dn_12bit", keys=many.keys)
    assert single.mean.shape == (1, 1)
    with pytest.raises(QueryAlignmentError, match="query points"):
        model.score(single, raw_many)


# --------------------------------------------------------------------------------------
# fix round 2: units / scale contract
# --------------------------------------------------------------------------------------


def test_predict_inverts_the_declared_unit_affine() -> None:
    """mean and variance leave predict in the query's declared units (variance × scale²)."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    history = make_history()
    identity = make_query(points=((0.5, 0.5), (1.5, 0.5)))
    affine = QuerySpec(
        keys=identity.keys,
        sensor_coordinates=identity.sensor_coordinates,
        camera_ids=identity.camera_ids,
        frame_index=identity.frame_index,
        units=identity.units,
        unit_affine={"cam0": UnitAffine(scale=2.5, offset=100.0, units=identity.units)},
    )
    plain = model.predict(world, history, identity, call="fit")
    scaled = model.predict(world, history, affine, call="fit")
    assert torch.allclose(scaled.mean, plain.mean * 2.5 + 100.0, rtol=0, atol=0)
    assert torch.allclose(scaled.variance, plain.variance * 2.5 * 2.5, rtol=1e-6)
    assert float(scaled.masks["unit_scale"][0, 0]) == 2.5
    assert float(scaled.masks["unit_offset"][0, 0]) == 100.0
    assert bool(scaled.masks["units_declared"].all())


def test_unit_affine_contract_rejects_the_internal_prediction_domain() -> None:
    """A normalized/internal-domain prediction must not be scored against DN units."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    history = make_history()
    spec = QuerySpec(
        keys=make_query().keys,
        sensor_coordinates=make_query().sensor_coordinates,
        camera_ids=make_query().camera_ids,
        frame_index=1,
        units="dn_12bit",
        unit_affine={"cam0": UnitAffine(scale=4031.0, offset=64.0, units="dn_12bit")},
    )
    raw = RawQuery(
        values=torch.full((1, spec.num_points), 2000.0),
        masks={"valid": torch.ones(1, spec.num_points, dtype=torch.bool)},
        units="dn_12bit",
        keys=spec.keys,
        camera_index=spec.camera_index(),
        unit_affine=spec.unit_affine,
    )
    declared = model.predict(world, history, spec, call="fit")
    assert model.score(declared, raw) is not None

    # the raw decoder output (internal domain, no unit declaration) is refused
    internal = FakeRawPrediction(
        mean=torch.full((1, spec.num_points), 0.5),
        variance=torch.full((1, spec.num_points), 1e-6),
        masks={"valid": torch.ones(1, spec.num_points, dtype=torch.bool)},
        units="dn_12bit",
    )
    with pytest.raises(QueryAlignmentError, match="unit transform"):
        model.score(internal, raw)

    # without the declaration the same internal values are an identifiable 1e6+ NLL
    plain_spec = make_query(points=((0.0, 0.0),) * 3)
    plain_raw = RawQuery(
        values=torch.full((1, plain_spec.num_points), 2000.0),
        units="dn_12bit",
        keys=plain_spec.keys,
    )
    plain_internal = FakeRawPrediction(
        mean=torch.full((1, plain_spec.num_points), 0.5),
        variance=torch.full((1, plain_spec.num_points), 1e-6),
        masks={},
        units="dn_12bit",
    )
    huge = model.score(plain_internal, plain_raw)
    assert float(huge.raw_nll_or_energy.mean()) > 1e5


def test_analytic_oracle_in_declared_units_scores_o1_with_finite_gradients() -> None:
    """The scale-consistency probe: true DN mean and DN² variance give an O(1) NLL."""
    from owac.frameworks.n02_full import sensor as sensor_mod

    sensor_model = sensor_mod.SensorModel(units="dn_12bit", density_verified=True)
    mean = torch.full((1, 3), 2048.0, requires_grad=True)
    variance = torch.full((1, 3), 4.0)  # declared noise variance in DN²
    oracle = AnalyticOracle(
        lambda history, spec: (mean, variance),
        prediction_type=sensor_mod.RawPrediction,
        sensor_model=sensor_model,
    )
    spec = make_query(points=((0.5, 0.5), (1.5, 0.5), (2.5, 0.5)))
    prediction = oracle.predict(oracle.world(), make_history(), spec, call="fit")
    deviations = torch.tensor([[1.5, -2.0, 0.5]])
    raw = RawQuery(values=mean.detach() + deviations, units=spec.units, keys=spec.keys)
    score = oracle.score(prediction, raw)
    per_point = float(score.per_valid_scalar.detach())
    assert math.isfinite(per_point)
    assert per_point < 10.0
    score.per_valid_scalar.backward()
    assert mean.grad is not None
    assert bool(torch.isfinite(mean.grad).all())


def test_world_vector_parts_share_one_device_and_dtype() -> None:
    model = make_model()
    world = CandidateSpec(
        "correct",
        "correct",
        CandidateGeometry(world_px=TRUTH.clone()),
        nuisance={"log_exposure": 0.25, "camera_shift_px": (1.0, -2.0), "noise_scale": 1.5},
    ).as_world()
    vector = model._world_vector(world)
    assert vector.shape == (model.candidate_dim,)
    assert vector.dtype == torch.float32
    assert vector.device == world.entities[0].geometry["world_px"].device


# --------------------------------------------------------------------------------------
# bounded third round: the camera-index comparison across devices
# --------------------------------------------------------------------------------------

_AVAILABLE_DEVICES = [torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize("left_device", _AVAILABLE_DEVICES)
@pytest.mark.parametrize("right_device", _AVAILABLE_DEVICES)
def test_camera_index_guard_compares_across_devices(left_device, right_device) -> None:
    """Equal indices pass and different indices fail, whatever device each was built on."""
    model = make_model(sensor_shape={"cam0": (4, 4), "cam1": (4, 4)})
    world = make_dictionary().worlds()[0]
    history = make_history()
    pair = QuerySpec(
        keys=spec_keys_two_cameras(),
        sensor_coordinates=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        camera_ids=("cam0", "cam1"),
        frame_index=2,
        units="dn_12bit",
    )
    prediction = model.predict(world, history, pair.select_camera("cam0"), call="fit")
    prediction.masks["camera_index"] = prediction.masks["camera_index"].to(left_device)

    matching = RawQuery(
        values=torch.zeros(1, 1),
        masks={"valid": torch.ones(1, 1, dtype=torch.bool)},
        units="dn_12bit",
        keys=pair.keys,
        camera_index=torch.zeros(1, 1, dtype=torch.long, device=right_device),
    )
    assert model.score(prediction, matching) is not None  # equal -> no misreport

    different = RawQuery(
        values=torch.zeros(1, 1),
        masks={"valid": torch.ones(1, 1, dtype=torch.bool)},
        units="dn_12bit",
        keys=pair.keys,
        camera_index=torch.ones(1, 1, dtype=torch.long, device=right_device),
    )
    with pytest.raises(QueryAlignmentError, match="camera"):
        model.score(prediction, different)  # different -> still the same error type


def spec_keys_two_cameras() -> tuple[MeasurementKey, ...]:
    return (
        MeasurementKey("ep0", "cam0", "cam0-f0002", "cam0-f0002"),
        MeasurementKey("ep0", "cam1", "cam1-f0002", "cam1-f0002"),
    )


def test_indices_match_helper_is_device_agnostic_and_exact() -> None:
    left = torch.tensor([[0, 0, 1]], dtype=torch.long)
    assert _indices_match(left, left.clone())
    assert not _indices_match(left, torch.tensor([[0, 1, 1]], dtype=torch.long))
    expanded = torch.zeros(1, 3, dtype=torch.long).expand(1, 3)
    assert _indices_match(expanded, torch.zeros(1, 3, dtype=torch.long))


# --------------------------------------------------------------------------------------
# hot-path performance guards (full-resolution queries; the old per-point Python loops
# made one predict cost ~0.16 s at Q=131072, i.e. tens of seconds per measurement step)
# --------------------------------------------------------------------------------------


def test_normalized_coordinates_match_the_legacy_per_point_implementation_bitwise() -> None:
    """Vectorized normalization keeps the exact arithmetic order of the old code."""
    model = make_model(sensor_shape={"cam0": (4, 4), "cam1": (6, 8)})
    spec = QuerySpec(
        keys=spec_keys_two_cameras(),
        sensor_coordinates=torch.tensor([[0.5, 0.5], [3.5, 3.5], [5.5, 7.5]], dtype=torch.float32),
        camera_ids=("cam0", "cam0", "cam1"),
        frame_index=2,
        units="dn_12bit",
    )
    shift = torch.tensor([[[1.0, -2.0], [0.5, 0.5], [-1.5, 3.0]]])
    for batch in (1, 4):
        for use_shift in (None, shift):
            expected = _legacy_normalized_coordinates(model, spec, use_shift, batch)
            actual = model._normalized_coordinates(spec, use_shift, batch)
            assert torch.equal(actual, expected), f"batch={batch} shift={use_shift is not None}"


def _legacy_normalized_coordinates(model, query_spec, shift, batch) -> torch.Tensor:
    """The pre-vectorization implementation, kept as the numerical reference."""
    raw = query_spec.sensor_coordinates.to(dtype=torch.float32)
    if shift is None:
        raw = raw.unsqueeze(0).expand(batch, -1, -1)
    else:
        raw = raw.unsqueeze(0) - shift
    result = torch.empty_like(raw)
    offsets = [index for index in range(len(query_spec.camera_ids))]
    for camera_id in dict.fromkeys(query_spec.camera_ids):
        height, width = model.sensor_shapes[camera_id]
        index = [i for i in offsets if query_spec.camera_ids[i] == camera_id]
        result[:, index, 0] = 2.0 * raw[:, index, 0] / float(height) - 1.0
        result[:, index, 1] = 2.0 * raw[:, index, 1] / float(width) - 1.0
    return result


def _large_query(*, per_camera: int = 65536, cameras: int = 2) -> QuerySpec:
    keys = tuple(
        MeasurementKey("ep0", f"cam{index}", "cam0-f0002", "cam0-f0002") for index in range(cameras)
    )
    coordinates = torch.rand(cameras * per_camera, 2) * 512.0
    camera_ids = tuple(key.camera_id for key in keys for _ in range(per_camera))
    return QuerySpec(
        keys=keys,
        sensor_coordinates=coordinates,
        camera_ids=camera_ids,
        frame_index=2,
        units="dn_12bit",
        unit_affine={
            key.camera_id: UnitAffine(scale=4031.0, offset=64.0, units="dn_12bit") for key in keys
        },
    )


def _paired_medians(first, second, repeats: int = 3) -> tuple[float, float]:
    """Median wall time of two callables measured alternately.

    Torch intra-op threads are pinned to one for the measurement: a loaded host
    oversubscribes them and would otherwise punish the tensor path more than the
    single-threaded legacy loop, hiding the real speed-up.
    """
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
    middle = repeats // 2
    return first_times[middle], second_times[middle]


def test_predict_large_query_hot_path_is_vectorized() -> None:
    """One full-resolution predict must beat the legacy per-point loop.

    Both sides are measured alternately in one process, so the comparison blocks the
    old implementation without making a busy host fail the suite.
    """
    torch.manual_seed(5)
    model = make_model(sensor_shape={"cam0": (512, 512), "cam1": (512, 512)})
    spec = _large_query()
    points = spec.num_points
    history = make_history(batch=8)
    world = make_dictionary().worlds()[0]
    model.predict(world, history, spec, call="fit")  # warm the caches once
    vectorized, legacy = _paired_medians(
        lambda: model.predict(world, history, spec, call="fit"),
        lambda: _legacy_normalized_coordinates(model, spec, None, 8),
    )
    assert vectorized < 3.0, f"predict took {vectorized:.3f}s for Q={points}"
    assert vectorized < 0.9 * legacy, (
        f"vectorized predict {vectorized:.4f}s is not faster than the legacy per-point "
        f"loop {legacy:.4f}s at Q={points}"
    )


def test_step_bundle_large_query_hot_path_is_vectorized() -> None:
    """The whole measurement step beats the legacy per-candidate cost it used to pay."""
    torch.manual_seed(5)
    side = 128
    model = make_model(
        sensor_shape={"cam0": (side, side), "cam1": (side, side)},
    )
    model.bind_dictionary(make_dictionary())
    batch = 8
    episodes = tuple(f"ep{index}" for index in range(batch))
    values = torch.full((batch, 1, side, side), 2000.0)
    masks = torch.ones(batch, side, side, dtype=torch.bool)
    frame = MeasurementFrame(
        values=values,
        valid_mask=masks,
        bad_mask=~masks,
        saturated_mask=~masks,
        episode_ids=episodes,
        frame_ids=tuple(f"{episode}-f0002" for episode in episodes),
        camera_id="cam0",
        new_measurement=torch.ones(batch, dtype=torch.bool),
        metadata={
            "unit": "raw_dn_uint16_mosaic",
            "calibration": {"black_level": 64.0, "white_level": 4095.0, "bit_depth": 12},
        },
        metadata_available={"calibration": True},
    )
    bundle = ObservationBundle(
        frames={"cam0": frame},
        context=EmbodimentContext(),
        episode_ids=episodes,
        anchor_ids=tuple(f"{episode}#anchor" for episode in episodes),
        new_time_mask=torch.ones(batch, dtype=torch.bool),
    )
    history = HistoryContext(
        pooled_features=torch.full((batch, FEATURE_DIM), 0.5),
        levels=None,
        episode_ids=episodes,
        anchor_ids=bundle.anchor_ids,
        last_frame_index=1,
        eta=None,
    )
    step_bundle(bundle, SequenceState(), model, history, ledger=MeasurementLedger())
    spec = _row_query_spec(bundle, model, batch=batch)
    step_time, legacy_predict = _paired_medians(
        lambda: step_bundle(bundle, SequenceState(), model, history, ledger=MeasurementLedger()),
        lambda: _legacy_normalized_coordinates(model, spec, None, batch),
    )
    rows = batch
    worlds = len(model.candidates.worlds())
    legacy_step_estimate = legacy_predict * 2 * rows * worlds
    assert step_time < 6.0, f"step took {step_time:.3f}s"
    assert step_time < 0.9 * legacy_step_estimate, (
        f"vectorized step {step_time:.4f}s is not faster than the legacy estimate "
        f"{legacy_step_estimate:.4f}s"
    )


def _row_query_spec(bundle, model, batch: int = 1) -> QuerySpec:
    from owac.frameworks.n02_full.model import _build_row_query

    row_query = _build_row_query(
        bundle,
        row=0,
        batch=batch,
        episode_id="ep0",
        history=make_history(),
        stride=int(model.config.query_stride),
        prediction_domain="units",
        config=model.config,
    )
    return row_query.spec


def test_world_vector_is_built_once_per_world_object(monkeypatch) -> None:
    """The measurement loop predicts K worlds per row: the conditioning vector is cached."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query()
    history = make_history()
    calls = {"count": 0}
    original = ObservationModel._build_world_vector

    def counted(self, item):
        calls["count"] += 1
        return original(self, item)

    monkeypatch.setattr(ObservationModel, "_build_world_vector", counted)
    for _ in range(5):
        model.predict(world, history, spec, call="fit")
    assert calls["count"] == 1


def test_predict_runs_no_device_wide_finite_check(monkeypatch) -> None:
    """``isfinite().all()`` synchronizes CUDA; the per-candidate hot path must not call it."""
    model = make_model()
    world = make_dictionary().worlds()[0]
    spec = make_query()
    history = make_history()
    calls = {"count": 0}
    original = torch.isfinite

    def counted(value):
        calls["count"] += 1
        return original(value)

    monkeypatch.setattr(torch, "isfinite", counted)
    model.predict(world, history, spec, call="fit")
    assert calls["count"] == 0


def test_camera_index_tensor_is_shared_between_prediction_and_query() -> None:
    """Prediction masks and the raw query use one cached index: no device comparison."""
    from owac.frameworks.n02_full.model import _build_raw_query, _build_row_query

    model = make_model(sensor_shape={"cam0": (4, 4), "cam1": (4, 4)})
    bundle = ObservationBundle(
        frames={
            "cam0": MeasurementFrame(
                values=torch.zeros(4, 4),
                valid_mask=torch.ones(4, 4, dtype=torch.bool),
                bad_mask=torch.zeros(4, 4, dtype=torch.bool),
                saturated_mask=torch.zeros(4, 4, dtype=torch.bool),
                episode_ids=("ep0",),
                frame_ids=("cam0-f0002",),
                camera_id="cam0",
                metadata={"unit": "dn_12bit"},
            ),
            "cam1": MeasurementFrame(
                values=torch.zeros(6, 8),
                valid_mask=torch.ones(6, 8, dtype=torch.bool),
                bad_mask=torch.zeros(6, 8, dtype=torch.bool),
                saturated_mask=torch.zeros(6, 8, dtype=torch.bool),
                episode_ids=("ep0",),
                frame_ids=("cam1-f0002",),
                camera_id="cam1",
                metadata={"unit": "dn_12bit"},
            ),
        },
        context=EmbodimentContext(),
        episode_ids=("ep0",),
        anchor_ids=("ep0#anchor",),
        new_time_mask=torch.tensor([True]),
    )
    history = make_history()
    row_query = _build_row_query(
        bundle,
        row=0,
        batch=1,
        episode_id="ep0",
        history=history,
        stride=2,
        prediction_domain="units",
        config=model.config,
    )
    raw = _build_raw_query(bundle, row_query, row=0, batch=1)
    prediction = model.predict(make_dictionary().worlds()[0], history, row_query.spec, call="fit")
    prediction_index = prediction.masks["camera_index"].reshape(-1)
    assert prediction_index.data_ptr() == raw.camera_index.reshape(-1).data_ptr()
    assert model.score(prediction, raw) is not None


def test_real_encoder_batched_rows_equal_per_row_encoding_bitwise() -> None:
    """One batched forward must give every row exactly the features a solo forward gives."""
    from owac.frameworks.n02_full.observation import RawObservationEncoder
    from owac.frameworks.n02_full.state import ObservationBundle

    torch.manual_seed(11)
    encoder = RawObservationEncoder(
        input_mode="raw3", depths=(1, 1, 1, 1), dims=(8, 16, 32, 64), projection_dim=8
    )
    height, width = 16, 16
    rows = 4
    values = torch.stack(
        [torch.full((3, height, width), float(index + 1)) for index in range(rows)]
    )
    masks = torch.ones(rows, height, width, dtype=torch.bool)

    def make_bundle(selected) -> ObservationBundle:
        from owac.frameworks.n02_full.state import MeasurementFrame

        frame = MeasurementFrame(
            values=values[selected],
            valid_mask=masks[selected],
            bad_mask=~masks[selected],
            saturated_mask=~masks[selected],
            episode_ids=tuple(f"ep{index}" for index in selected),
            frame_ids=tuple(f"ep{index}-f0002" for index in selected),
            camera_id="cam0",
            new_measurement=torch.ones(len(selected), dtype=torch.bool),
            metadata={"unit": "dn_12bit"},
        )
        return ObservationBundle(
            frames={"cam0": frame},
            context=EmbodimentContext(),
            episode_ids=tuple(f"ep{index}" for index in selected),
            anchor_ids=tuple(f"ep{index}#anchor" for index in selected),
            new_time_mask=torch.ones(len(selected), dtype=torch.bool),
        )

    batched = encoder.encode(make_bundle(list(range(rows))))
    batched_pooled = pool_feature_pyramid(batched)
    assert batched_pooled.shape[0] == rows
    worst = 0.0
    for index in range(rows):
        solo = encoder.encode(make_bundle([index]))
        solo_pooled = pool_feature_pyramid(solo)
        # conv kernels reduce in a different order for batch>1, so equality is to
        # float32 noise (~1e-9) rather than bitwise; nothing statistical is introduced.
        worst = max(
            worst, float((batched_pooled[index : index + 1] - solo_pooled).abs().max().detach())
        )
        assert torch.allclose(
            batched_pooled[index : index + 1], solo_pooled, rtol=0.0, atol=1e-6
        ), index
    assert worst < 1e-6, f"batched vs per-row encoder delta {worst}"

    # slicing a batched encode is exactly a per-row pool
    selected_pooled = pool_feature_pyramid(batched, rows=[1, 2])
    assert torch.equal(selected_pooled, batched_pooled[1:3])


def test_raw_query_validates_indices_once_and_rejects_bad_ones() -> None:
    """Validation still applies (one reduction), it is only skipped for trusted instances."""
    key_a = MeasurementKey("ep0", "cam0", "f1", "f1")
    key_b = MeasurementKey("ep0", "cam1", "f1", "f1")
    with pytest.raises(ValueError, match="camera_index"):
        RawQuery(
            values=torch.zeros(2),
            units="dn",
            keys=(key_a, key_b),
            camera_index=torch.tensor([2, 0]),
        )
    with pytest.raises(ValueError, match="camera_index"):
        RawQuery(
            values=torch.zeros(2),
            units="dn",
            keys=(key_a, key_b),
            camera_index=torch.tensor([-1, 0]),
        )
    good = RawQuery(
        values=torch.zeros(2),
        units="dn",
        keys=(key_a, key_b),
        camera_index=torch.tensor([0, 1]),
    )
    assert good.camera_point_counts() == {"cam0": 1, "cam1": 1}
    # the trusted fast path is for our own already-validated constructions
    trusted = RawQuery._trusted(
        values=torch.zeros(2),
        masks={},
        units="dn",
        keys=(key_a, key_b),
        camera_index=torch.tensor([0, 1]),
    )
    assert torch.equal(trusted.camera_index, good.camera_index)

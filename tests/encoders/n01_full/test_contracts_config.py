"""Behavioural tests for the N01 v1.0 contracts and configuration.

Each test names the wrong implementation it is able to reject.  The oracle is
either an independent construction (a hand-written identity tuple, a literal
JSON document) or a property the contract states, never the module echoing
itself.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import pytest

from owac.frameworks.n01_full.config import (
    REPLAY_MODES,
    BasisConfig,
    ContentConfig,
    CorrespondenceConfig,
    HistoryConfig,
    InputIdentityConfig,
    N01FullConfig,
    ReplayConfig,
    TrainingBudgetConfig,
)
from owac.frameworks.n01_full.contracts import (
    VERSION_FIELDS,
    CalibratedMeasurement,
    CorrespondenceMap,
    DependencyNode,
    EvidenceEstimate,
    MeasurementRecord,
    ReadoutSnapshot,
    ReplayRange,
    SourceKey,
    SupportDomain,
)


def _key(
    *,
    dataset: str = "ds0",
    episode: str = "ep0",
    camera: str = "cam0",
    frame_id: int | str = 7,
    phase: int | str = 0,
    y: int = 3,
    x: int = 5,
) -> SourceKey:
    return SourceKey(
        dataset_id=dataset,
        episode_id=episode,
        camera_id=camera,
        frame_id=frame_id,
        channel_or_phase=phase,
        sensor_y=y,
        sensor_x=x,
    )


def _record(
    *,
    frame_id: int | str = 7,
    raw_value: float | int | None = 512,
    validity_kind: str = "valid",
    calibration_version: str = "cal-v1",
    **key_overrides: object,
) -> MeasurementRecord:
    return MeasurementRecord(
        key=_key(frame_id=frame_id, **key_overrides),
        raw_value=raw_value,
        value_units="raw_counts",
        timestamp=frame_id if isinstance(frame_id, int) else 0,
        calibration_version=calibration_version,
        validity_kind=validity_kind,
    )


def _domain(
    *,
    domain_id: str = "d0",
    evidence: tuple[SourceKey, ...] = (),
    context: tuple[SourceKey, ...] = (),
) -> SupportDomain:
    return SupportDomain(
        domain_id=domain_id,
        reference_chart="chart-0",
        basis="phase_local_affine",
        phase=0,
        evidence_source_ids=evidence,
        temporal_segment_id="seg-0",
        decision_version="dec-v1",
        mapping_version="map-v1",
        context_source_ids=context,
    )


def _versions() -> dict[str, str]:
    return {
        "parameter_version": "p1",
        "support_version": "s1",
        "mapping_version": "m1",
        "calibration_version": "c1",
    }


# --- SourceKey identity -----------------------------------------------------


def test_value_identical_exposures_are_distinct_sources() -> None:
    """An ID that hashes the sample value collides here; frame identity does not."""
    first_exposure = _record(frame_id=11, raw_value=512)
    second_exposure = _record(frame_id=12, raw_value=512)
    assert first_exposure.raw_value == second_exposure.raw_value == 512
    assert first_exposure.key != second_exposure.key
    assert len({first_exposure.key, second_exposure.key}) == 2


def test_source_key_identity_cannot_be_partially_declared() -> None:
    """No default field means no ID can be built from a value plus a batch slot alone."""
    for item in dataclasses.fields(SourceKey):
        assert item.default is dataclasses.MISSING
        assert item.default_factory is dataclasses.MISSING
    assert _key(y=3, x=5).identity() == ("ds0", "ep0", "cam0", 7, 0, 3, 5)
    with pytest.raises(ValueError, match="sensor index"):
        _key(y=-1)
    with pytest.raises(ValueError, match="non-empty identity string"):
        _key(camera="")


def test_equal_values_at_different_sensor_positions_do_not_collide() -> None:
    """A value-keyed implementation returns one ID for these two different samples."""
    here = _record(y=3, x=5, raw_value=512).key
    elsewhere = _record(y=3, x=6, raw_value=512).key
    assert here != elsewhere


def test_repeated_packet_collapses_to_one_source_id() -> None:
    """Registering every transmission separately would make this length two."""
    first = _key(frame_id=7, phase=1, y=9, x=2)
    repeated = _key(frame_id=7, phase=1, y=9, x=2)
    assert first == repeated
    assert hash(first) == hash(repeated)
    assert len(dict.fromkeys([first, repeated, first])) == 1
    assert len({first, repeated}) == 1


def test_camera_episode_phase_and_dataset_do_not_collide() -> None:
    """Dropping any identity field from the key merges samples that share a frame id."""
    base = _key()
    neighbours = (
        _key(camera="cam1"),
        _key(episode="ep1"),
        _key(dataset="ds1"),
        _key(phase=1),
        _key(frame_id=8),
        _key(y=4),
        _key(x=6),
    )
    assert len({base, *neighbours}) == 1 + len(neighbours)


# --- MeasurementRecord semantics -------------------------------------------


def test_rejection_is_a_support_relation_state_not_a_sensor_state() -> None:
    """Writing "rejected" into the sensor record is refused; leaving the domain is not."""
    with pytest.raises(ValueError, match="support relation"):
        _record(validity_kind="rejected", raw_value=4095)
    record = _record(validity_kind="saturated", raw_value=4095)
    listed = _domain(evidence=(record.key,))
    retracted = _domain(domain_id="d1", evidence=())
    assert listed.role_of(record.key) == "evidence"
    assert retracted.role_of(record.key) is None
    assert record.validity_kind == "saturated"
    assert record.raw_value == 4095


def test_missing_measurement_is_not_a_zero_observation() -> None:
    """Filling a gap with 0.0 would silently add a fake observation."""
    missing = _record(validity_kind="missing", raw_value=None)
    assert missing.raw_value is None
    with pytest.raises(ValueError, match="not carry a value"):
        _record(validity_kind="missing", raw_value=0)
    with pytest.raises(ValueError, match="requires a measured raw_value"):
        _record(validity_kind="valid", raw_value=None)


def test_raw_value_is_immutable_and_calibration_is_a_versioned_derivative() -> None:
    """Rewriting raw history in place fails; re-calibrating creates a new object."""
    record = _record(raw_value=100, calibration_version="cal-v1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.raw_value = 0.0  # type: ignore[misc]
    recalibrated = CalibratedMeasurement.from_record(record, 25.0, calibration_version="cal-v2")
    assert recalibrated.raw_value == record.raw_value == 100
    assert recalibrated.calibration_version == "cal-v2"
    assert record.calibration_version == "cal-v1"
    with pytest.raises(ValueError, match="no calibrated value"):
        CalibratedMeasurement.from_record(_record(validity_kind="missing", raw_value=None), 1.0)


# --- CorrespondenceMap ------------------------------------------------------


def test_correspondence_taps_stay_inside_the_mapped_source_frame() -> None:
    """Taps from another frame, unmatched weights or taps on an invalid map are refused."""
    inside = _key(frame_id=4)
    outside = _key(frame_id=5)
    mapping = CorrespondenceMap(
        source_frame_id=4,
        reference_frame_id=7,
        source_xy=(3.5, 5.25),
        tap_source_ids=(inside,),
        tap_weights=(1.0,),
        valid=True,
        mapping_version="map-v1",
        visibility=True,
        flow_risk=0.0,
    )
    assert mapping.tap_count == 1
    with pytest.raises(ValueError, match="not from the mapped source frame"):
        CorrespondenceMap(
            source_frame_id=4,
            reference_frame_id=7,
            source_xy=(3.5, 5.25),
            tap_source_ids=(outside,),
            tap_weights=(1.0,),
            valid=True,
            mapping_version="map-v1",
        )
    with pytest.raises(ValueError, match="exactly one source sample and one weight"):
        CorrespondenceMap(
            source_frame_id=4,
            reference_frame_id=7,
            source_xy=(3.5, 5.25),
            tap_source_ids=(inside,),
            tap_weights=(),
            valid=True,
            mapping_version="map-v1",
        )
    with pytest.raises(ValueError, match="carries no sampling taps"):
        CorrespondenceMap(
            source_frame_id=4,
            reference_frame_id=7,
            source_xy=(3.5, 5.25),
            tap_source_ids=(inside,),
            tap_weights=(1.0,),
            valid=False,
            mapping_version="map-v1",
        )


# --- SupportDomain roles ----------------------------------------------------


def test_evidence_and_context_roles_are_kept_separate() -> None:
    """Sharing one list, or double-listing a source, would mix support with context."""
    evidence = _key(frame_id=1)
    context = _key(frame_id=2)
    domain = _domain(evidence=(evidence,), context=(context,))
    assert domain.role_of(evidence) == "evidence"
    assert domain.role_of(context) == "context"
    assert domain.evidence_count == 1 and domain.context_count == 1
    with pytest.raises(ValueError, match="both evidence and context"):
        _domain(evidence=(evidence,), context=(evidence,))
    with pytest.raises(ValueError, match="duplicate source"):
        _domain(evidence=(evidence, evidence))


def test_evidence_sources_may_appear_in_several_candidate_domains() -> None:
    """A shared source across domains is allowed; the estimate counts it once."""
    shared = _key(frame_id=1)
    domain_a = _domain(domain_id="a", evidence=(shared,))
    domain_b = _domain(domain_id="b", evidence=(shared, _key(frame_id=2)))
    assert domain_a.role_of(shared) == "evidence"
    assert domain_b.role_of(shared) == "evidence"


# --- EvidenceEstimate -------------------------------------------------------


def test_prior_is_not_counted_as_an_independent_observation() -> None:
    """len(ids) gives 3 and len(ids)+prior gives 4; the contract wants 2."""
    first, second = _key(frame_id=1), _key(frame_id=2)
    estimate = EvidenceEstimate(
        theta=(1.0, 2.0),
        uncertainty_kind="covariance",
        identifiable_mask=(True, True),
        evidence_source_ids=(first, second, second),
        support_version="s1",
        covariance=((0.5, 0.0), (0.0, 0.5)),
        prior_id="prior-v1",
    )
    assert estimate.independent_observation_count == 2
    assert estimate.has_prior
    without_prior = dataclasses.replace(estimate, prior_id=None)
    assert without_prior.independent_observation_count == 2
    assert not without_prior.has_prior


def test_rank_deficiency_is_reported_not_regularized_away() -> None:
    """An unavailable covariance must not smuggle a point estimate in as precision."""
    identifiers = _key(frame_id=1)
    with pytest.raises(ValueError, match="rank deficiency must be reported"):
        EvidenceEstimate(
            theta=(1.0,),
            uncertainty_kind="none",
            identifiable_mask=None,
            evidence_source_ids=(identifiers,),
            support_version="s1",
        )
    with pytest.raises(ValueError, match="requires a covariance"):
        EvidenceEstimate(
            theta=(1.0,),
            uncertainty_kind="covariance",
            identifiable_mask=(True,),
            evidence_source_ids=(identifiers,),
            support_version="s1",
        )
    with pytest.raises(ValueError, match="must not carry a covariance"):
        EvidenceEstimate(
            theta=(1.0,),
            uncertainty_kind="none",
            identifiable_mask=(False,),
            evidence_source_ids=(identifiers,),
            support_version="s1",
            covariance=((1.0,),),
        )


# --- Dependency and readout -------------------------------------------------


def test_parameter_update_retires_derived_dependency_nodes() -> None:
    """Comparing only support_version, or ignoring unknown keys, fails here."""
    node = DependencyNode(
        node_id="feature-0",
        kind="feature",
        parent_ids=("measurement-0",),
        parameter_version="p1",
        support_version="s1",
        mapping_version="m1",
        calibration_version="c1",
    )
    assert node.versions() == _versions()
    assert not node.is_stale_against(node.versions())
    assert node.is_stale_against({"parameter_version": "p2"})
    assert node.is_stale_against({"calibration_version": "c2"})
    with pytest.raises(ValueError, match="Unknown version fields"):
        node.is_stale_against({"bogus_version": "x"})


def test_committed_readout_carries_all_versions_range_and_cost() -> None:
    """A snapshot that cannot name its versions is not a commit."""
    snapshot = ReadoutSnapshot(
        representation="tokens",
        validity=True,
        source_time=10,
        commit_time=11,
        versions=_versions(),
        exact_replay_range=ReplayRange(oldest_frame_id=1, newest_frame_id=32, unique_frames=32),
        cost={"updates": 1, "ms": 4.5},
    )
    assert snapshot.version_of("support_version") == "s1"
    assert snapshot.exact_replay_range.covers(32)
    assert not snapshot.exact_replay_range.covers(33)
    assert snapshot.versions == _versions()
    with pytest.raises(TypeError):
        snapshot.versions["support_version"] = "s2"  # type: ignore[index]
    with pytest.raises(ValueError, match="dependency version"):
        dataclasses.replace(snapshot, versions={"parameter_version": "p1"})
    with pytest.raises(ValueError, match="unique frame"):
        ReplayRange(oldest_frame_id=1, newest_frame_id=1, unique_frames=0)
    assert set(VERSION_FIELDS) == set(_versions())


# --- Configuration ----------------------------------------------------------


def test_config_defaults_follow_the_shared_contract() -> None:
    """Wrong defaults (history 4, exact window 8, CHW layout, C=3 CFA) fail here."""
    config = N01FullConfig()
    assert config.history.max_history == 8
    assert config.replay.exact_window_unique_frames == 32
    assert config.input.batch_layout == "BTCHW"
    assert config.input.time_order == "oldest_first"
    assert config.content.fusion_time_order == "newest_first"
    assert config.input.input_mode == "rawvla_rgb3"
    assert config.input.channels == 3 and config.input.cfa_pattern is None
    assert config.correspondence.coordinate_convention == "pull"
    assert config.content.readout_tokens == 256
    assert config.content.readout_width == 2048
    assert config.content.bridge_in_dim == config.content.feature_width == 512
    assert config.content.bridge_width == config.content.readout_width == 2048
    assert (
        set(config.replay.modes)
        == set(REPLAY_MODES)
        == {
            "fixed_decision_replay",
            "counterfactual_replay",
        }
    )
    cfa = N01FullConfig(input=InputIdentityConfig.bayer_cfa("sensor-x"))
    assert cfa.input.channels == 4 and cfa.input.cfa_pattern == "RGGB"
    assert cfa.input.is_cfa and not config.input.is_cfa


def test_config_refuses_the_known_wrong_conventions() -> None:
    """Each case is a documented mistake: push coordinates, fake CFA, one replay mode."""
    with pytest.raises(ValueError, match="pull coordinate"):
        CorrespondenceConfig(coordinate_convention="push").validate()
    with pytest.raises(ValueError, match="not this convention"):
        CorrespondenceConfig.from_dict(CorrespondenceConfig(coordinate_convention="push").to_dict())
    with pytest.raises(ValueError, match="RGB-space intermediate"):
        N01FullConfig(input=InputIdentityConfig(input_mode="rgb", channels=3))
    with pytest.raises(ValueError, match="no CFA phase"):
        N01FullConfig(input=InputIdentityConfig(input_mode="rawvla_rgb3", cfa_pattern="RGGB"))
    with pytest.raises(ValueError, match="four CFA phases"):
        N01FullConfig(input=InputIdentityConfig(input_mode="bayer_cfa", channels=3))
    with pytest.raises(ValueError, match="both replay interfaces"):
        N01FullConfig(replay=ReplayConfig(modes=("fixed_decision_replay",)))
    with pytest.raises(ValueError, match="8/16/32 unique frames"):
        N01FullConfig(replay=ReplayConfig(exact_window_unique_frames=64))
    with pytest.raises(ValueError, match="strictly increasing unique windows"):
        N01FullConfig(history=HistoryConfig(comparison_windows=(8, 8, 32)))
    with pytest.raises(ValueError, match="exact replay window"):
        N01FullConfig(
            history=HistoryConfig(max_history=32),
            replay=ReplayConfig(exact_window_unique_frames=8),
        )
    with pytest.raises(ValueError, match="preserve_cfa_phase"):
        N01FullConfig(
            input=InputIdentityConfig.bayer_cfa("sensor-x"),
            correspondence=CorrespondenceConfig(preserve_cfa_phase=False),
        )
    with pytest.raises(ValueError, match="whole number of"):
        N01FullConfig(correspondence=CorrespondenceConfig(input_channels=5))
    with pytest.raises(ValueError, match="equal the content feature width"):
        N01FullConfig(content=ContentConfig(bridge_in_dim=2048))
    with pytest.raises(ValueError, match="prior_id must be None"):
        N01FullConfig(basis=BasisConfig(prior_kind="none", prior_id="prior-v1"))
    with pytest.raises(ValueError, match="requires an explicit prior_id"):
        N01FullConfig(basis=BasisConfig(prior_kind="gaussian"))
    with pytest.raises(ValueError, match="separate streams"):
        TrainingBudgetConfig(model_seeds=(101, 211, 307), data_seeds=(101,)).validate()
    with pytest.raises(ValueError, match="at least three model seeds"):
        TrainingBudgetConfig(model_seeds=(101, 211)).validate()


def test_to_dict_matches_a_hand_written_document() -> None:
    """The oracle is a literal document, so a dropped field cannot pass silently."""
    config = N01FullConfig(
        input=InputIdentityConfig.bayer_cfa("sensor-x", cfa_pattern="BGGR"),
        history=HistoryConfig(max_history=16, comparison_windows=(8, 16)),
        budget=TrainingBudgetConfig(model_seeds=(101, 211, 307), data_seeds=(9001,)),
    )
    document = config.to_dict()
    assert document["config_version"] == "n01-full-v1"
    assert document["input"] == {
        "dataset_id": "sensor-x",
        "input_mode": "bayer_cfa",
        "channels": 4,
        "cfa_pattern": "BGGR",
        "value_units": "raw_integer_counts",
        "coordinate_units": "sensor_pixel",
        "batch_layout": "BTCHW",
        "time_order": "oldest_first",
    }
    assert document["history"] == {
        "max_history": 16,
        "comparison_windows": [8, 16],
        "summary_keeps_provenance": True,
    }
    assert document["replay"]["exact_window_unique_frames"] == 32
    assert json.loads(json.dumps(document)) == document


def test_config_json_round_trip_is_lossless_and_rejects_unknown_fields() -> None:
    """Silently dropping an unknown field, or losing tuple/nested types, fails here."""
    config = N01FullConfig(
        input=InputIdentityConfig.bayer_cfa("sensor-x"),
        correspondence=CorrespondenceConfig(mapping_kind="sea_raft", input_channels=8),
        budget=TrainingBudgetConfig(model_seeds=(101, 211, 307), data_seeds=(9001, 9002)),
    )
    restored = N01FullConfig.from_json(config.to_json())
    assert restored == config
    assert restored.to_dict() == config.to_dict()
    assert restored.fingerprint() == config.fingerprint()
    assert isinstance(restored.history.comparison_windows, tuple)
    assert isinstance(restored.input, InputIdentityConfig)

    payload = config.to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="Unknown N01FullConfig fields"):
        N01FullConfig.from_dict(payload)
    nested = config.to_dict()
    nested["content"]["mystery"] = 1
    with pytest.raises(ValueError, match="Unknown ContentConfig fields"):
        N01FullConfig.from_dict(nested)
    with pytest.raises(ValueError, match="Unknown TrainingBudgetConfig fields"):
        N01FullConfig.from_dict(
            {**config.to_dict(), "budget": {**config.to_dict()["budget"], "mystery": 1}}
        )
    changed = N01FullConfig.from_dict({**config.to_dict(), "config_version": "other"})
    assert changed.fingerprint() != config.fingerprint()


def test_importing_the_package_does_not_import_torch() -> None:
    """A module-level torch import (or an implicit device init) fails this subprocess."""
    program = (
        "import sys; import owac.frameworks.n01_full; print('torch=' + str('torch' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "torch=False"

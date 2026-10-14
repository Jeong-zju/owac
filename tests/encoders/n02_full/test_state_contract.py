"""State, identity and evidence-accounting tests for the N02 full (v1.1) line.

Every assertion here is a semantic one from the shared contract: a repeated
acquisition adds no evidence, the same source frame re-acquired is a separate
measurement, a calibration replacement never adds evidence, the objective key is
order independent over calibration versions, a partial reset leaves the other
rows bit for bit unchanged, and stage verdicts are restricted.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from owac.core.encoder_contract import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full import state as state_mod


def _key(episode: str = "ep0", camera: str = "cam", frame: str = "f0", acq: str = "a0"):
    return state_mod.MeasurementKey(episode, camera, frame, acq)


def _ledger_with(*keys) -> state_mod.MeasurementLedger:
    ledger = state_mod.MeasurementLedger()
    for key in keys:
        assert ledger.register(
            state_mod.MeasurementRecord(key=key, raw_ref=f"{key.acquisition_id}.npz")
        )
    return ledger


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_measurement_key_hash_is_stable_and_identity_sensitive() -> None:
    key = _key()
    assert key.key_hash() == _key().key_hash()
    assert len(key.key_hash()) == 64
    assert key.key_hash() != _key(acq="a1").key_hash()
    assert key.key_hash() != _key(camera="cam2").key_hash()
    assert key.key_hash() != _key(episode="ep1").key_hash()
    assert key.key_hash() != _key(frame="f1").key_hash()
    assert key.as_dict() == {
        "episode_id": "ep0",
        "camera_id": "cam",
        "source_frame_id": "f0",
        "acquisition_id": "a0",
    }


def test_measurement_key_requires_acquisition_identity() -> None:
    with pytest.raises(ValueError):
        _key(acq="")


def test_measurement_set_hash_is_order_and_duplicate_insensitive() -> None:
    first = _key(acq="a0")
    second = _key(acq="a1")
    third = _key(episode="ep1", frame="f9", acq="a2")
    forward = state_mod.measurement_set_hash([first, second, third])
    assert len(forward) == 64
    assert state_mod.measurement_set_hash([third, first, second]) == forward
    assert state_mod.measurement_set_hash([first, second, third, first]) == forward
    assert state_mod.measurement_set_hash([first, second]) != forward
    assert state_mod.measurement_set_hash([]) == state_mod.measurement_set_hash([])


# ---------------------------------------------------------------------------
# ledger: no duplicated evidence
# ---------------------------------------------------------------------------


def test_duplicate_acquisition_adds_no_evidence() -> None:
    ledger = state_mod.MeasurementLedger()
    key = _key()
    record = state_mod.MeasurementRecord(key=key, raw_ref="frame.npz")
    assert ledger.register(record) is True
    version_after_first = ledger.state_version
    assert ledger.register(record) is False
    assert ledger.register(state_mod.MeasurementRecord(key=key, raw_ref="retransmit.npz")) is False
    assert ledger.evidence_count == 1
    assert len(ledger) == 1
    assert ledger.state_version == version_after_first
    assert ledger.get(key).raw_ref == "frame.npz"


def test_same_source_frame_with_new_acquisition_counts_once_more() -> None:
    ledger = _ledger_with(_key(acq="a0"), _key(acq="a1"))
    assert ledger.evidence_count == 2
    assert ledger.state_version == 2
    assert {key.acquisition_id for key in ledger.keys()} == {"a0", "a1"}


def test_distinct_episodes_are_distinct_evidence() -> None:
    ledger = _ledger_with(_key(), _key(episode="ep1", acq="a1"))
    assert ledger.evidence_count == 2
    assert ledger.has(_key(episode="ep1", acq="a1"))
    assert not ledger.has(_key(episode="ep2"))
    assert ledger.get(_key(episode="ep2")) is None
    assert ledger.get(_key(episode="ep2"), "missing") == "missing"


# ---------------------------------------------------------------------------
# ledger: calibration replacement
# ---------------------------------------------------------------------------


def test_replace_calibration_keeps_evidence_count_and_raises_for_unknown_key() -> None:
    key = _key()
    ledger = _ledger_with(key)
    before = ledger.get(key).as_dict()
    assert ledger.replace_calibration(key, "cal-v2") is True
    assert ledger.evidence_count == 1
    assert len(ledger) == 1
    assert ledger.state_version == 2
    after = ledger.get(key)
    assert after.calibration_version == "cal-v2"
    assert after.raw_ref == before["raw_ref"]
    assert after.masks == before["masks"]
    assert after.noise_metadata == before["noise_metadata"]
    assert after.key == key

    assert ledger.replace_calibration(key, "cal-v2") is False
    assert ledger.state_version == 2
    with pytest.raises(KeyError):
        ledger.replace_calibration(_key(acq="unknown"), "cal-v3")
    with pytest.raises(KeyError):
        ledger.replace_calibration(_key(episode="ep9"), "cal-v3")


# ---------------------------------------------------------------------------
# ledger: reset and serialization
# ---------------------------------------------------------------------------


def test_reset_clears_only_the_named_episodes() -> None:
    kept_a = _key(acq="a0")
    kept_b = _key(episode="ep1", frame="f1", acq="a1")
    dropped = _key(episode="ep2", frame="f2", acq="a2")
    ledger = _ledger_with(kept_a, kept_b, dropped)
    kept_before = {key.acquisition_id: ledger.get(key).as_dict() for key in (kept_a, kept_b)}
    version_before = ledger.state_version
    removed = ledger.reset("ep2")

    assert removed == 1
    assert ledger.evidence_count == 2
    assert ledger.state_version == version_before + 1
    assert not ledger.has(dropped)
    for key in (kept_a, kept_b):
        assert ledger.get(key).as_dict() == kept_before[key.acquisition_id]

    records_before = [record.as_dict() for record in ledger.records()]
    assert ledger.reset(["ep-absent"]) == 0
    assert ledger.reset("ep-absent") == 0
    assert ledger.state_version == version_before + 1
    assert [record.as_dict() for record in ledger.records()] == records_before


def test_reset_accepts_several_episodes_at_once() -> None:
    ledger = _ledger_with(
        _key(acq="a0"), _key(episode="ep1", acq="a1"), _key(episode="ep2", acq="a2")
    )
    assert ledger.reset(["ep0", "ep2"]) == 2
    assert [key.episode_id for key in ledger.keys()] == ["ep1"]


def test_ledger_json_round_trip_is_equal() -> None:
    ledger = _ledger_with(_key(acq="a0"), _key(episode="ep1", acq="a1"))
    ledger.replace_calibration(_key(episode="ep1", acq="a1"), "cal-v9")
    ledger.reset("ep0")
    text = ledger.to_json()
    restored = state_mod.MeasurementLedger.from_json(text)
    assert restored == ledger
    assert restored.state_version == ledger.state_version
    assert restored.evidence_count == ledger.evidence_count
    assert restored.to_json() == text
    assert state_mod.MeasurementLedger.from_json(ledger.as_dict()) == ledger

    fresh = state_mod.MeasurementLedger()
    assert state_mod.MeasurementLedger.from_json(fresh.to_json()) == fresh


def test_ledger_state_version_counts_only_successful_mutations() -> None:
    ledger = state_mod.MeasurementLedger()
    assert ledger.state_version == 0
    key = _key()
    ledger.register(state_mod.MeasurementRecord(key=key))
    assert ledger.state_version == 1
    ledger.register(state_mod.MeasurementRecord(key=key))
    assert ledger.state_version == 1
    ledger.replace_calibration(key, "v1")
    assert ledger.state_version == 2
    ledger.replace_calibration(key, "v1")
    assert ledger.state_version == 2
    ledger.reset("other")
    assert ledger.state_version == 2
    ledger.reset("ep0")
    assert ledger.state_version == 3


# ---------------------------------------------------------------------------
# objective key
# ---------------------------------------------------------------------------


def test_objective_key_normalizes_and_is_order_independent() -> None:
    first = state_mod.ObjectiveKey("set-hash", "model-v1", "structure-v1", ["c2", "c1"], "prior-v1")
    second = state_mod.ObjectiveKey(
        "set-hash", "model-v1", "structure-v1", ("c1", "c2"), "prior-v1"
    )
    assert first.calibration_versions == ("c1", "c2")
    assert first.key_hash() == second.key_hash()
    assert first == second
    assert (
        first.key_hash()
        != state_mod.ObjectiveKey(
            "set-hash", "model-v1", "structure-v1", ("c1",), "prior-v1"
        ).key_hash()
    )
    assert (
        first.key_hash()
        != state_mod.ObjectiveKey(
            "set-hash", "model-v1", "structure-v1", ("c1", "c2"), "prior-v2"
        ).key_hash()
    )
    assert first.as_dict()["calibration_versions"] == ["c1", "c2"]


def test_objective_key_rejects_a_non_sequence_version_field() -> None:
    with pytest.raises(TypeError):
        state_mod.ObjectiveKey("set-hash", "model-v1", "structure-v1", 7)


# ---------------------------------------------------------------------------
# container identity
# ---------------------------------------------------------------------------


def test_observation_bundle_keeps_camera_identity() -> None:
    frames = {
        "cam_left": MeasurementFrame(values=[1.0], camera_id="cam_left"),
        "cam_right": MeasurementFrame(values=[2.0], camera_id="cam_right"),
    }
    bundle = state_mod.ObservationBundle(
        frames=frames,
        context=EmbodimentContext(),
        episode_ids=["ep0"],
        anchor_ids=("a0",),
        new_time_mask=[True],
    )
    assert bundle.camera_ids == ("cam_left", "cam_right")
    assert bundle.episode_ids == ("ep0",)
    assert bundle.anchor_ids == ("a0",)
    assert bundle.new_time_mask == [True]
    inference = state_mod.InferenceInput(bundle, declared_prior=None)
    assert inference.bundle is bundle

    with pytest.raises(TypeError):
        state_mod.ObservationBundle(frames=[], context=EmbodimentContext())
    with pytest.raises(TypeError):
        state_mod.ObservationBundle(frames={1: frames["cam_left"]}, context=EmbodimentContext())


def test_simple_state_containers_coerce_their_identity_fields() -> None:
    level = state_mod.FeatureLevel(
        values=[1.0], valid=[True], sensor_coordinates=[[0, 0]], support_ids=["s0"], level_id="l0"
    )
    assert level.support_ids == ("s0",)
    entity = state_mod.EntityState(entity_ids=["e0"], geometry_frame="view_2d", version=3)
    assert entity.entity_ids == ("e0",)
    assert entity.version == 3
    targets = state_mod.TrainingTargets(actions=[0.0], action_mask=[True])
    assert targets.optional_geometry is None
    world = state_mod.WorldHypothesis(
        world_id="w0", entities=("e0",), score=1.5, score_kind="empirical_energy"
    )
    assert world.entities == ("e0",)
    assert world.ancestor_id is None
    assert world.version == 0
    assert world.score_kind == "empirical_energy"


@pytest.mark.parametrize("kind", ["normalized_density", "empirical_energy", "importance_weight"])
def test_world_hypothesis_accepts_declared_score_kinds(kind: str) -> None:
    assert state_mod.WorldHypothesis("w0", score_kind=kind).score_kind == kind


def test_world_hypothesis_rejects_an_undeclared_score_kind() -> None:
    with pytest.raises(ValueError) as excinfo:
        state_mod.WorldHypothesis("w0", score_kind="softmax")
    assert "softmax" in str(excinfo.value)


# ---------------------------------------------------------------------------
# partial-batch reset
# ---------------------------------------------------------------------------


def test_reset_batch_resets_only_selected_rows() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.arange(12.0).reshape(3, 4)
    existence = torch.ones(3, dtype=torch.bool)
    state = state_mod.SequenceState(
        episode_ids=("e0", "e1", "e2"),
        slot_hidden=hidden,
        existence=existence,
        geometry=torch.arange(3.0),
        version=7,
    )
    row1_hidden = state.slot_hidden[1].tolist()
    row1_geometry = state.geometry[1].item()

    new = state.reset_batch([True, False, True])

    assert new.version == 8
    assert new.episode_ids == ("", "e1", "")
    assert new.slot_hidden[1].tolist() == row1_hidden
    assert new.geometry[1].item() == row1_geometry
    assert new.slot_hidden[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert new.slot_hidden[2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert new.existence[0].item() is False or bool(new.existence[0]) is False
    assert bool(new.existence[1]) is True
    assert state.version == 7
    assert state.slot_hidden[0].tolist() == [0.0, 1.0, 2.0, 3.0]

    tensor_mask = torch.tensor([False, True, False])
    from_tensor = state.reset_batch(tensor_mask)
    assert from_tensor.episode_ids == ("e0", "", "e2")
    assert from_tensor.slot_hidden[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert from_tensor.slot_hidden[1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_reset_batch_all_false_returns_an_equal_copy() -> None:
    torch = pytest.importorskip("torch")
    state = state_mod.SequenceState(
        episode_ids=("e0", "e1"),
        slot_hidden=torch.ones(2, 3),
        existence=torch.ones(2, dtype=torch.bool),
        version=4,
    )
    copy = state.reset_batch([False, False])
    assert copy == state
    assert copy.version == 4
    assert copy.slot_hidden is not state.slot_hidden
    assert copy.episode_ids == state.episode_ids
    copy.slot_hidden[0, 0] = 9.0
    assert state.slot_hidden[0, 0].item() == 1.0


def test_reset_batch_rejects_a_mask_of_the_wrong_length() -> None:
    state = state_mod.SequenceState(episode_ids=("e0", "e1"), slot_hidden=[[0.0], [0.0]])
    with pytest.raises(ValueError) as excinfo:
        state.reset_batch([True, False, True])
    assert "length" in str(excinfo.value)
    with pytest.raises(ValueError):
        state.reset_batch([True])
    with pytest.raises(TypeError):
        state.reset_batch(None)


def test_reset_batch_works_with_plain_python_payloads() -> None:
    state = state_mod.SequenceState(
        episode_ids=("e0", "e1"),
        slot_hidden=[[1.0, 2.0], [3.0, 4.0]],
        existence=[True, True],
        geometry=[0.5, 0.25],
        version=1,
    )
    new = state.reset_batch([False, True])
    assert new.episode_ids == ("e0", "")
    assert new.slot_hidden == [[1.0, 2.0], [0, 0]]
    assert new.existence == [True, False]
    assert new.geometry == [0.5, 0]
    assert new.version == 2


def test_reset_batch_infers_batch_size_from_the_payload_when_ids_are_empty() -> None:
    state = state_mod.SequenceState(slot_hidden=[[0.0], [0.0]], version=2)
    assert state.batch_size == 2
    assert state.reset_batch([True, False]).version == 3
    with pytest.raises(ValueError):
        state.reset_batch([True, False, False])


# ---------------------------------------------------------------------------
# stage verdicts
# ---------------------------------------------------------------------------


def test_stage_result_records_four_independent_axes() -> None:
    result = state_mod.StageResult(
        implementation="incomplete",
        numerical="pass",
        behavior="not_run",
        closed_loop="not_run",
        evidence_paths=["outputs/n02-full/p0/report/report.json"],
        notes={"source": "unit"},
    )
    payload = result.as_dict()
    assert payload["implementation"] == "incomplete"
    assert payload["numerical"] == "pass"
    assert payload["behavior"] == "not_run"
    assert payload["closed_loop"] == "not_run"
    assert payload["evidence_paths"] == ["outputs/n02-full/p0/report/report.json"]
    assert payload["notes"] == {"source": "unit"}
    assert set(state_mod.STAGE_VERDICTS) == {"pass", "fail", "not_run", "incomplete"}


@pytest.mark.parametrize("axis", ["implementation", "numerical", "behavior", "closed_loop"])
@pytest.mark.parametrize("value", ["passed", "ok", "", None, "PASS"])
def test_stage_result_rejects_an_illegal_verdict(axis: str, value: object) -> None:
    kwargs = {
        "implementation": "not_run",
        "numerical": "not_run",
        "behavior": "not_run",
        "closed_loop": "not_run",
    }
    kwargs[axis] = value
    with pytest.raises(ValueError) as excinfo:
        state_mod.StageResult(**kwargs)
    assert axis in str(excinfo.value)


# ---------------------------------------------------------------------------
# light import path
# ---------------------------------------------------------------------------


def test_state_module_does_not_import_torch_or_numpy() -> None:
    script = (
        "import sys; import owac.frameworks.n02_full.state as s; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'numpy' not in sys.modules, 'numpy imported'; "
        "print(s.MeasurementKey('e', 'c', 'f', 'a').key_hash())"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip()) == 64

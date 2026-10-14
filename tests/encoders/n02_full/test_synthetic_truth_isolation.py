"""Truth-isolation tests for the N02 full synthetic generator.

The contract under test is that an inference payload contains RAW measurements,
declared calibration, available embodiment context, language tokens, executed
actions and a declared prior only. Truth and future supervision live in separate
objects: injecting sentinels into them may not change anything an inference call
can read, and a future action may not change a past measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from owac.frameworks.n02_full import synthetic as synth

FRAMES = 12
CAMERA = "cam0"


def sensors() -> tuple[synth.SensorSpec, ...]:
    return (
        synth.SensorSpec(
            camera_id=CAMERA,
            height=48,
            width=64,
            input_mode="bayer_cfa",
            cfa_pattern="RGGB",
            black_level=64.0,
            white_level=4095.0,
            bit_depth=12,
            read_noise_e=2.0,
        ),
    )


def sequence(seed: int = 21, *, horizon: int = FRAMES, action_horizon: int = 6, actions=None):
    scene = synth.generate_scene("train", "isolation-0001", canvas=(64, 64), horizon=horizon)
    schedule = synth.build_event_schedule(
        scene, frames=horizon, seed=synth.schedule_seed("train", "isolation-0001"), camera_id=CAMERA
    )
    return synth.generate_sequence(
        scene,
        sensors(),
        synth.MeasurementModel(kind="cfa"),
        schedule,
        frames=horizon,
        seed=seed,
        actions=actions,
        action_horizon=action_horizon,
    )


def arrays_of(value, out: set) -> None:
    """Collect ids of every numpy array reachable from a payload."""
    if isinstance(value, np.ndarray):
        out.add(id(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            arrays_of(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            arrays_of(item, out)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for spec in dataclasses.fields(value):
            arrays_of(getattr(value, spec.name), out)


def inference_digest(payloads) -> str:
    """Content hash of everything reachable from the inference payloads."""
    digest = hashlib.sha256()

    def walk(value, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            if array.dtype.kind == "f":
                digest.update(np.nan_to_num(array, nan=1.0e30).tobytes())
            else:
                digest.update(array.tobytes())
            return
        if isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                walk(value[key], depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for spec in dataclasses.fields(value):
                digest.update(spec.name.encode("ascii"))
                walk(getattr(value, spec.name), depth + 1)
            return
        digest.update(repr(value).encode("utf-8"))

    for payload in payloads:
        walk(payload)
    return digest.hexdigest()


def forbidden_keys(value, path: str = "") -> list[str]:
    """Keys that would indicate truth, future supervision or an RGB image."""
    banned = ("truth", "future", "rgb", "srgb", "label", "object_id", "evaluation")
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(word in lowered for word in banned):
                found.append(f"{path}/{key}")
            found.extend(forbidden_keys(item, f"{path}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(forbidden_keys(item, f"{path}[{index}]"))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for spec in dataclasses.fields(value):
            found.extend(forbidden_keys(getattr(value, spec.name), f"{path}.{spec.name}"))
    return found


# --------------------------------------------------------------------------------------
# declared content of an inference payload
# --------------------------------------------------------------------------------------


def test_inference_payload_has_no_truth_future_or_rgb_fields() -> None:
    seq = sequence()
    assert len(seq.inference) == len(seq.frames) == len(seq.truth) == len(seq.targets) == FRAMES
    assert forbidden_keys(seq.inference) == []

    frame = seq.frames[0][CAMERA]
    assert "object_ids" not in frame.metadata
    for value in frame.metadata.values():
        assert "truth" not in str(value).lower()

    object_ids = set(seq.truth[0].object_ids)
    for payload in seq.inference:
        digest = hashlib.sha256()
        for measurement in payload.bundle.frames.values():
            empty = set()
            arrays_of(measurement, empty)
            digest.update(repr(sorted(measurement.metadata)).encode())
        assert object_ids  # the truth does carry identities, the payload does not
        for object_id in object_ids:
            assert object_id not in repr(payload.bundle.frames[CAMERA].metadata)


def test_inference_context_is_causally_available_only() -> None:
    seq = sequence()
    first = seq.inference[0].bundle.context
    assert first.executed_action is None
    assert first.executed_action_available is False
    assert first.proprio_available is True
    assert first.camera_transform is None
    assert first.camera_transform_available is False
    assert first.hypothesis_prior_available is True
    assert first.hypothesis_prior.sum() == pytest.approx(1.0)
    assert first.task_tokens.dtype == np.int64
    assert first.task_mask.all()

    second = seq.inference[1].bundle.context
    assert second.executed_action_available is True
    assert second.executed_action is not None
    prior = seq.inference[0].declared_prior
    assert prior["kind"] == "declared_uniform"
    assert sum(prior["mass"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# sentinels and aliasing
# --------------------------------------------------------------------------------------


def test_truth_and_target_sentinels_do_not_change_inference() -> None:
    seq = sequence()
    before = inference_digest(seq.inference)
    for truth in seq.truth:
        for array in truth.geometry.values():
            array[...] = 12345
        truth.existence[...] = True
        for array in truth.occlusion_mask.values():
            array[...] = True
        for array in truth.visible_fraction.values():
            array[...] = 0.5
    for target in seq.targets:
        for array in (
            target.actions,
            target.action_mask,
            target.optional_geometry,
            target.optional_relations,
        ):
            array[...] = 99
    assert inference_digest(seq.inference) == before
    assert synth.measurement_hash(seq) == synth.measurement_hash(sequence())


def test_inference_arrays_are_not_aliased_with_truth_or_targets() -> None:
    seq = sequence()
    inference: set = set()
    truth: set = set()
    targets: set = set()
    for payload in seq.inference:
        arrays_of(payload, inference)
    for item in seq.truth:
        arrays_of(item, truth)
    for item in seq.targets:
        arrays_of(item, targets)
    assert inference and truth and targets
    assert not inference & truth
    assert not inference & targets
    assert not truth & targets


def test_targets_carry_supervision_copies_not_truth_references() -> None:
    seq = sequence()
    target = seq.targets[0]
    truth = seq.truth[0]
    assert np.array_equal(target.optional_geometry, truth.geometry["world_px"])
    assert target.optional_geometry is not truth.geometry["world_px"]
    adjacency = target.optional_relations
    assert adjacency.shape == (len(truth.object_ids), len(truth.object_ids))
    assert np.array_equal(adjacency, adjacency.T)
    assert target.actions.shape == (6, 2)
    assert target.action_mask.dtype == bool


# --------------------------------------------------------------------------------------
# causality
# --------------------------------------------------------------------------------------


def test_future_actions_do_not_change_past_inference() -> None:
    """The measured past and its inference payload are independent of later actions."""
    horizon, switch = 12, 7
    past = np.zeros((horizon + 6, 2), dtype=np.int8)
    altered = past.copy()
    altered[switch:] = 2

    base = sequence(seed=23, horizon=horizon, action_horizon=6, actions=past)
    changed = sequence(seed=23, horizon=horizon, action_horizon=6, actions=altered)

    for step in range(switch + 1):
        assert np.array_equal(
            base.frames[step][CAMERA].values, changed.frames[step][CAMERA].values
        ), step
        assert np.array_equal(
            base.inference[step].bundle.context.executed_action if step > 0 else np.zeros(2),
            changed.inference[step].bundle.context.executed_action if step > 0 else np.zeros(2),
        ), step
    for step in range(switch + 1, horizon):
        assert (
            base.frames[step][CAMERA].values.tobytes()
            != changed.frames[step][CAMERA].values.tobytes()
        ), step
    assert synth.measurement_hash(base) != synth.measurement_hash(changed)
    assert not np.array_equal(base.targets[switch].actions, changed.targets[switch].actions)
    assert np.array_equal(
        base.targets[switch - 1].actions[:1], changed.targets[switch - 1].actions[:1]
    )


def test_executed_action_history_is_strictly_past() -> None:
    actions = np.arange(2 * (FRAMES + 6), dtype=np.int8).reshape(FRAMES + 6, 2)
    seq = sequence(seed=24, horizon=FRAMES, action_horizon=6, actions=actions)
    for step in range(FRAMES):
        context = seq.inference[step].bundle.context
        if step == 0:
            assert context.executed_action is None
            continue
        assert np.array_equal(context.executed_action, actions[step - 1].astype(np.float32))
        expected_future = np.zeros((6, 2), dtype=np.float32)
        for offset in range(6):
            if step + offset < FRAMES:
                expected_future[offset] = actions[step + offset]
        assert np.array_equal(seq.targets[step].actions, expected_future)
        # the current action is supervision only; it never appears as executed context
        assert not np.array_equal(context.executed_action, actions[step].astype(np.float32))

    assert seq.targets[FRAMES - 1].action_mask.sum() == 1
    assert not seq.targets[FRAMES - 1].action_mask[1:].any()
    assert seq.targets[0].action_mask.all()


def test_anchor_ids_are_unique_and_new_time_mask_is_declared() -> None:
    seq = sequence()
    anchors = [payload.bundle.anchor_ids[0] for payload in seq.inference]
    assert len(set(anchors)) == FRAMES
    for payload in seq.inference:
        assert payload.bundle.episode_ids == (payload.bundle.anchor_ids[0].split("#")[0],)
        assert bool(np.asarray(payload.bundle.new_time_mask).all())
        assert payload.bundle.frames[CAMERA].is_new_measurement
        assert payload.bundle.new_time_mask.shape == (1,)

"""Noise-free generator hook tests for the P1 oracle upper bound (T2, P0 extension).

``MeasurementModel.mean_variance(sensor, state)`` must return the expectation of one
camera's measurement in the same declared unit as ``render()``, must reuse the renderer's
own projection/response/exposure/noise-parameter path, and must not touch any random
stream.  These tests pin the unit semantics against the renderer, prove bit-level RNG
invariance, and check the expectation against Monte-Carlo noise draws.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from owac.frameworks.n02_full import synthetic as synth

SHOT = 1000.0
BACKGROUND = 0.4
READ_E = 2.0
DRAWS = 400


def cfa_sensor(*, read_noise_e: float = READ_E) -> synth.SensorSpec:
    return synth.SensorSpec(
        camera_id="cam0",
        height=8,
        width=8,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
        black_level=64.0,
        white_level=4095.0,
        bit_depth=12,
        read_noise_e=read_noise_e,
        gain=1.0,
        exposure_ms=1.0,
    )


def cfa_model(*, read_noise_e: float = READ_E) -> synth.MeasurementModel:
    return synth.MeasurementModel(
        kind="cfa",
        shot_noise_scale=SHOT,
        read_noise_e=read_noise_e,
        saturation_level=1.0,
        response=(0.95, 1.0, 1.0, 0.75),
        bad_pixel_fraction=0.0,
        missing_sample_fraction=0.0,
    )


def raw3_sensor() -> synth.SensorSpec:
    return synth.SensorSpec(
        camera_id="cam0",
        height=6,
        width=6,
        input_mode="rawvla_rgb3",
        black_level=0.0,
        white_level=4095.0,
        bit_depth=16,
        read_noise_e=READ_E,
        gain=1.0,
        exposure_ms=1.0,
    )


def raw3_model() -> synth.MeasurementModel:
    return synth.MeasurementModel(
        kind="raw3",
        shot_noise_scale=SHOT,
        read_noise_e=READ_E,
        saturation_level=1.0,
        response=(0.9, 1.0, 0.7),
        bad_pixel_fraction=0.0,
        missing_sample_fraction=0.0,
    )


def flat_state(
    sensor: synth.SensorSpec,
    *,
    seed: int,
    background: float = BACKGROUND,
    exposure: float = 1.0,
    read_noise_e: float = READ_E,
) -> synth.RenderState:
    return synth.RenderState(
        frame_index=0,
        shapes=(),
        sensors=(sensor,),
        background_level=background,
        illumination=1.0,
        exposure_scale=exposure,
        shot_noise_scale=SHOT,
        read_noise_e=read_noise_e,
        read_noise_e_source="sensor_read_noise_e",
        camera_shift_px={sensor.camera_id: (0.0, 0.0)},
        noise_seed=seed,
    )


def test_mean_variance_reuses_the_renderer_unit_and_declaration():
    sensor = cfa_sensor()
    model = cfa_model()
    state = flat_state(sensor, seed=17)
    measurement = model.render(state)["cam0"]
    mean, variance = model.mean_variance(sensor, state)

    assert measurement.unit == model.unit() == "raw_dn_uint16_mosaic"
    assert mean.shape == measurement.values.shape == (8, 8)
    assert variance.shape == mean.shape
    assert mean.dtype == variance.dtype == np.float32
    assert np.all(mean > 0.0) and np.all(variance > 0.0)

    # independent re-derivation from the declared scene and sensor numbers
    plane = synth.cfa_pixel_plane_index(sensor.cfa_pattern, sensor.height, sensor.width)
    response = np.asarray(model.response, dtype=np.float32)
    mean_signal = response[plane] * BACKGROUND
    variance_signal = mean_signal / SHOT + (READ_E / SHOT) ** 2
    expected_mean = sensor.black_level + mean_signal * sensor.dn_scale
    expected_variance = variance_signal * sensor.dn_scale**2
    assert np.allclose(mean, expected_mean, rtol=1e-6)
    assert np.allclose(variance, expected_variance, rtol=1e-6)
    # the flat field stays far from the saturation clip, so the expectation is linear
    assert int(np.asarray(measurement.saturated_mask).sum()) == 0


def test_mean_variance_covers_the_three_channel_raw3_mode():
    sensor = raw3_sensor()
    model = raw3_model()
    state = flat_state(sensor, seed=3)
    measurement = model.render(state)["cam0"]
    mean, variance = model.mean_variance(sensor, state)
    assert measurement.unit == model.unit() == "pseudo_raw_dn_float32"
    assert mean.shape == measurement.values.shape == (3, 6, 6)
    response = np.asarray(model.response, dtype=np.float32)[:, None, None]
    mean_signal = response * BACKGROUND
    expected_variance = (mean_signal / SHOT + (READ_E / SHOT) ** 2) * sensor.dn_scale**2
    assert np.allclose(mean, mean_signal * sensor.dn_scale, rtol=1e-6)
    assert np.allclose(variance, expected_variance, rtol=1e-6)


def test_mean_variance_rejects_a_mode_mismatch_like_render():
    sensor = cfa_sensor()
    model = raw3_model()
    state = flat_state(sensor, seed=1)
    with pytest.raises(ValueError):
        model.mean_variance(sensor, state)
    with pytest.raises(ValueError):
        model.render(state)
    with pytest.raises(TypeError):
        cfa_model().mean_variance(object(), state)
    with pytest.raises(TypeError):
        cfa_model().mean_variance(sensor, object())


def _nan_aware_bytes(array: np.ndarray) -> bytes:
    array = np.asarray(array)
    if array.dtype.kind == "f":
        array = np.where(np.isnan(array), np.float32(-12345.0), array)
    return array.tobytes()


def _render_bytes(model: synth.MeasurementModel, state: synth.RenderState) -> list[bytes]:
    payload: list[bytes] = []
    for measurement in model.render(state).values():
        for name in (
            "values",
            "saturation_tail_probability",
            "valid_mask",
            "bad_mask",
            "missing_mask",
            "saturated_mask",
        ):
            payload.append(_nan_aware_bytes(getattr(measurement, name)))
    return payload


def test_hook_is_repeatable_and_never_advances_a_render_stream():
    sensor = cfa_sensor()
    model = cfa_model()
    first = model.mean_variance(sensor, flat_state(sensor, seed=11))
    second = model.mean_variance(sensor, flat_state(sensor, seed=11))
    assert np.array_equal(first[0], second[0], equal_nan=True)
    assert np.array_equal(first[1], second[1], equal_nan=True)

    # a seeded render sequence is bit-for-bit identical before and after hook calls
    def sequence() -> list[bytes]:
        payload: list[bytes] = []
        for frame_seed in (11, 12, 13, 14):
            payload.extend(_render_bytes(model, flat_state(sensor, seed=frame_seed)))
        return payload

    before = sequence()
    for _ in range(3):
        model.mean_variance(sensor, flat_state(sensor, seed=99))
    after = sequence()
    assert before == after


def test_hook_does_not_touch_the_global_numpy_random_state():
    sensor = cfa_sensor()
    model = cfa_model()
    np.random.seed(20261014)
    reference = [np.random.random(8).tobytes() for _ in range(4)]
    np.random.seed(20261014)
    model.mean_variance(sensor, flat_state(sensor, seed=5))
    model.mean_variance(sensor, flat_state(sensor, seed=6))
    repeated = [np.random.random(8).tobytes() for _ in range(4)]
    assert reference == repeated


def test_monte_carlo_draws_match_the_hook_expectation():
    sensor = cfa_sensor()
    model = cfa_model()
    mean, variance = model.mean_variance(sensor, flat_state(sensor, seed=0))
    sigma = np.sqrt(variance)

    samples = np.empty((DRAWS, *mean.shape), dtype=np.float64)
    for draw in range(DRAWS):
        measurement = model.render(flat_state(sensor, seed=1000 + draw))["cam0"]
        samples[draw] = measurement.values.astype(np.float64)

    sample_mean = samples.mean(axis=0)
    sample_variance = samples.var(axis=0, ddof=1)
    mean_z = (sample_mean - mean) / sigma
    variance_ratio = sample_variance / variance

    # per-pixel 4-sigma bounds, plus the far tighter pooled bounds
    assert np.all(np.abs(mean_z) <= 4.0 / np.sqrt(DRAWS)), np.abs(mean_z).max()
    assert np.all(np.abs(variance_ratio - 1.0) <= 4.0 * np.sqrt(2.0 / DRAWS)), np.abs(
        variance_ratio - 1.0
    ).max()
    pooled_mean_z = float(mean_z.mean())
    pooled_variance_ratio = float(variance_ratio.mean())
    pooled_tolerance = 4.0 / np.sqrt(DRAWS * mean.size)
    assert abs(pooled_mean_z) <= pooled_tolerance, pooled_mean_z
    assert abs(pooled_variance_ratio - 1.0) <= 4.0 * np.sqrt(2.0 / (DRAWS * mean.size))
    # keep the measured numbers visible in the failure message
    print(
        f"draws={DRAWS} pixels={mean.size} "
        f"max|mean_z|={float(np.abs(mean_z).max()):.4f} "
        f"max|var_ratio-1|={float(np.abs(variance_ratio - 1.0).max()):.4f} "
        f"pooled_mean_z={pooled_mean_z:.4f} pooled_var_ratio={pooled_variance_ratio:.4f}"
    )


# --------------------------------------------------------------------------------------
# capability 1: per-frame truth RenderState on the sequence
# --------------------------------------------------------------------------------------

HOOK_FRAMES = 4


def _quiet_sequence() -> tuple[synth.MeasurementModel, synth.SyntheticSequence]:
    scene = synth.generate_scene("train", "hook-0001", canvas=(32, 32), horizon=HOOK_FRAMES)
    sensor = cfa_sensor()
    model = cfa_model()
    schedule = synth.EventSchedule(horizon=HOOK_FRAMES, seed=2)
    sequence = synth.generate_sequence(
        scene, (sensor,), model, schedule, frames=HOOK_FRAMES, seed=42
    )
    return model, sequence


def test_sequence_keeps_the_exact_render_states_that_were_rendered():
    model, sequence = _quiet_sequence()
    assert len(sequence.render_states) == sequence.num_frames == len(sequence.frames)
    for frame_index, state in enumerate(sequence.render_states):
        assert isinstance(state, synth.RenderState)
        assert state.frame_index == frame_index
        assert state.sensors == sequence.cameras
        # the stored state is the rendered one: re-rendering it reproduces the frame
        rendered = model.render(state)["cam0"]
        stored = sequence.frames[frame_index]["cam0"]
        assert np.array_equal(rendered.values, stored.values)
        assert np.array_equal(
            rendered.saturation_tail_probability, stored.saturation_tail_probability
        )
        # and the declared world geometry agrees with the truth geometry of that frame
        truth = sequence.truth[frame_index]
        world = {
            object_id: tuple(position)
            for object_id, position in zip(
                truth.object_ids, np.asarray(truth.geometry["world_px"]).tolist(), strict=True
            )
        }
        for shape in state.shapes:
            if shape.shape_id in world:
                assert np.allclose(shape.position_px, world[shape.shape_id], atol=0.0)


def test_render_states_field_defaults_to_empty_for_older_sequences():
    field = {item.name: item for item in dataclasses.fields(synth.SyntheticSequence)}[
        "render_states"
    ]
    assert field.default == ()
    _, sequence = _quiet_sequence()
    legacy = synth.SyntheticSequence(
        sequence_id=sequence.sequence_id,
        scene_id=sequence.scene_id,
        seed=sequence.seed,
        partition=sequence.partition,
        cameras=sequence.cameras,
        frames=sequence.frames,
        inference=sequence.inference,
        targets=sequence.targets,
        truth=sequence.truth,
        manifest=sequence.manifest,
    )
    assert legacy.render_states == ()


def test_truth_hypothesis_from_render_states_reproduces_the_plain_hook():
    model, sequence = _quiet_sequence()
    sensor = sequence.cameras[0]
    state = sequence.render_states[1]
    truth_hypothesis = {
        shape.shape_id: {
            "world_px": shape.position_px,
            "scale_px": shape.scale_px,
            "depth": shape.depth,
        }
        for shape in state.shapes
    }
    plain_mean, plain_variance = model.mean_variance(sensor, state)
    truth_mean, truth_variance = model.mean_variance(sensor, state, hypothesis=truth_hypothesis)
    assert np.array_equal(plain_mean, truth_mean), "truth hypothesis must be a no-op"
    assert np.array_equal(plain_variance, truth_variance)


# --------------------------------------------------------------------------------------
# capability 2: geometry-hypothesis expectation
# --------------------------------------------------------------------------------------


def geometry_state() -> tuple[synth.SensorSpec, synth.MeasurementModel, synth.RenderState]:
    """Two declared discs well inside the frame, plus their frame state."""
    sensor = synth.SensorSpec(
        camera_id="cam0",
        height=48,
        width=48,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
        black_level=64.0,
        white_level=4095.0,
        bit_depth=12,
        read_noise_e=READ_E,
        gain=1.0,
        exposure_ms=1.0,
    )
    model = cfa_model()
    shapes = (
        synth.RenderShape(
            shape_id="obj_a",
            kind="disc",
            position_px=(12.0, 12.0),
            scale_px=4.0,
            colour_response=(0.9, 0.9, 0.9),
            depth=1,
        ),
        synth.RenderShape(
            shape_id="obj_b",
            kind="disc",
            position_px=(36.0, 36.0),
            scale_px=4.0,
            colour_response=(0.5, 0.5, 0.5),
            depth=2,
        ),
    )
    state = synth.RenderState(
        frame_index=0,
        shapes=shapes,
        sensors=(sensor,),
        background_level=0.2,
        illumination=1.0,
        exposure_scale=1.0,
        shot_noise_scale=SHOT,
        read_noise_e=READ_E,
        read_noise_e_source="sensor_read_noise_e",
        camera_shift_px={"cam0": (0.0, 0.0)},
        noise_seed=7,
    )
    return sensor, model, state


def intensity_centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity-weighted (row, column) centroid of a non-negative map."""
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    total = float(image.sum())
    assert total > 0.0
    return float((image * yy).sum() / total), float((image * xx).sum() / total)


def test_hypothesis_none_is_bit_identical_to_the_plain_hook():
    sensor, model, state = geometry_state()
    plain_mean, plain_variance = model.mean_variance(sensor, state)
    none_mean, none_variance = model.mean_variance(sensor, state, hypothesis=None)
    empty_mean, empty_variance = model.mean_variance(sensor, state, hypothesis={})
    assert np.array_equal(plain_mean, none_mean)
    assert np.array_equal(plain_variance, none_variance)
    assert np.array_equal(plain_mean, empty_mean)
    assert np.array_equal(plain_variance, empty_variance)


def test_hypothesis_moves_only_the_named_entity_by_the_declared_shift():
    sensor, model, state = geometry_state()
    truth_mean, truth_variance = model.mean_variance(sensor, state)
    # pixel-centre convention: the disc centred at (12, 12) has index centroid (11.5, 11.5)
    centre = (11.5, 11.5)
    for delta_y, delta_x in ((6, 9), (0, 8), (-4, 6)):
        shifted_mean, shifted_variance = model.mean_variance(
            sensor,
            state,
            hypothesis={"obj_a": {"world_px": (12.0 + delta_y, 12.0 + delta_x)}},
        )
        arrived = np.clip(shifted_mean - truth_mean, 0.0, None)
        left = np.clip(truth_mean - shifted_mean, 0.0, None)
        arrived_centre = intensity_centroid(arrived)
        left_centre = intensity_centroid(left)
        # the high-response region moved from the old to the new position (<= 1.5 px)
        assert abs(arrived_centre[0] - (centre[0] + delta_y)) <= 1.5, (delta_y, arrived_centre)
        assert abs(arrived_centre[1] - (centre[1] + delta_x)) <= 1.5, (delta_x, arrived_centre)
        assert abs(left_centre[0] - centre[0]) <= 1.0, left_centre
        assert abs(left_centre[1] - centre[1]) <= 1.0, left_centre
        # the displacement itself recovers the declared shift within rasterisation
        assert abs((arrived_centre[0] - left_centre[0]) - delta_y) <= 1.5, arrived_centre
        assert abs((arrived_centre[1] - left_centre[1]) - delta_x) <= 1.5, arrived_centre
        # the untouched entity keeps its patch bit for bit
        patch = (slice(30, 43), slice(30, 43))
        assert np.array_equal(truth_mean[patch], shifted_mean[patch])
        assert np.array_equal(truth_variance[patch], shifted_variance[patch])
    # an alias field name behaves the same way
    via_alias, _ = model.mean_variance(
        sensor, state, hypothesis={"obj_a": {"position_px": (18.0, 21.0)}}
    )
    alias_arrived = intensity_centroid(np.clip(via_alias - truth_mean, 0.0, None))
    assert abs(alias_arrived[0] - (centre[0] + 6.0)) <= 1.5, alias_arrived
    assert abs(alias_arrived[1] - (centre[1] + 9.0)) <= 1.5, alias_arrived
    # a scale override grows a response annulus around the same entity centre
    grown, _ = model.mean_variance(sensor, state, hypothesis={"obj_a": {"scale_px": 8.0}})
    annulus = np.clip(grown - truth_mean, 0.0, None)
    added_pixels = int((annulus > 1.0).sum())
    # scale 4 -> 8 adds pi*(8^2 - 4^2) ~= 151 sensor pixels of response
    assert added_pixels > 100, added_pixels
    annulus_centre = intensity_centroid(annulus)
    assert abs(annulus_centre[0] - centre[0]) <= 1.0, annulus_centre
    assert abs(annulus_centre[1] - centre[1]) <= 1.0, annulus_centre


def test_hypothesis_rejects_unknown_entities_and_non_geometry_fields():
    sensor, model, state = geometry_state()
    with pytest.raises(ValueError):
        model.mean_variance(sensor, state, hypothesis={"obj_z": {"world_px": (1.0, 2.0)}})
    with pytest.raises(ValueError):
        model.mean_variance(
            sensor, state, hypothesis={"obj_a": {"colour_response": (1.0, 0.0, 0.0)}}
        )
    with pytest.raises(ValueError):
        model.mean_variance(sensor, state, hypothesis={"obj_a": {"world_px": (1.0,)}})
    with pytest.raises(ValueError):
        model.mean_variance(sensor, state, hypothesis={"obj_a": {"scale_px": 0.0}})
    with pytest.raises(TypeError):
        model.mean_variance(sensor, state, hypothesis={"obj_a": 3.0})


def test_hypothesis_calls_do_not_advance_the_render_stream():
    sensor, model, state = geometry_state()

    def sequence() -> list[bytes]:
        return [
            _nan_aware_bytes(model.render(flat_state(sensor, seed=seed))["cam0"].values)
            for seed in (21, 22, 23)
        ]

    before = sequence()
    truth_hypothesis = {shape.shape_id: {"world_px": shape.position_px} for shape in state.shapes}
    for _ in range(3):
        model.mean_variance(sensor, state, hypothesis={"obj_a": {"world_px": (24.0, 24.0)}})
        model.mean_variance(sensor, state, hypothesis=truth_hypothesis)
    assert sequence() == before

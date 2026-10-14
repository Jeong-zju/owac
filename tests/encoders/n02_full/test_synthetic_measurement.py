"""Measurement-semantics tests for the N02 full synthetic generator.

The two declared observation models are checked as models: CFA plane phase and
coordinates against the RawVLA tools, declared units and dtypes against the stored
arrays, the noise/variance declaration against a measured flat-field variance, and
the saturation statement against a tail/interval probability. The cost probe is
checked for being a real measurement rather than a constant.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from owac.frameworks.n02_full import synthetic as synth
from owac.raw import rawvla_sequence as reference

FRAMES = 10


def cfa_sensor(camera_id: str = "cam0", *, pattern: str = "RGGB", read_noise_e: float = 0.0):
    return synth.SensorSpec(
        camera_id=camera_id,
        height=48,
        width=64,
        input_mode="bayer_cfa",
        cfa_pattern=pattern,
        black_level=64.0,
        white_level=4095.0,
        bit_depth=12,
        read_noise_e=read_noise_e,
        gain=1.0,
        exposure_ms=1.0,
    )


def raw3_sensor(camera_id: str = "cam3"):
    return synth.SensorSpec(
        camera_id=camera_id,
        height=48,
        width=64,
        input_mode="rawvla_rgb3",
        black_level=0.0,
        white_level=4095.0,
        bit_depth=16,
        read_noise_e=0.0,
        gain=1.0,
        exposure_ms=1.0,
    )


def background_module_scene(*, background: float = 0.4, shape_scale: float = 1.0):
    """A scene whose only structure sits far outside the frame: a flat field."""
    shape = synth.ShapeSpec(
        shape_id="far",
        kind="disc",
        scale_px=shape_scale,
        position_px=(-80.0, -80.0),
        velocity_px_per_step=(0.0, 0.0),
        colour_response=(0.0, 0.0, 0.0),
    )
    return synth.SceneSpec(
        scene_id="flat-field",
        seed=1,
        partition="train",
        shapes=(shape,),
        background_level=background,
        camera_motion_px=(0.0, 0.0),
    )


def quiet_schedule(horizon: int = FRAMES, seed: int = 2) -> synth.EventSchedule:
    return synth.EventSchedule(horizon=horizon, seed=seed)


# --------------------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------------------


def test_measurement_hash_is_reproducible_and_seed_sensitive() -> None:
    """Same seed reproduces every measured sample; a new seed does not."""
    scene = synth.generate_scene("train", "meas-0001", canvas=(64, 64), horizon=8)
    schedule = synth.build_event_schedule(
        scene, frames=8, seed=synth.schedule_seed("train", "meas-0001"), camera_id="cam0"
    )
    model = synth.MeasurementModel(kind="cfa")
    first = synth.generate_sequence(scene, (cfa_sensor(),), model, schedule, frames=8, seed=42)
    second = synth.generate_sequence(scene, (cfa_sensor(),), model, schedule, frames=8, seed=42)
    other = synth.generate_sequence(scene, (cfa_sensor(),), model, schedule, frames=8, seed=43)

    assert synth.measurement_hash(first) == synth.measurement_hash(second)
    assert synth.measurement_hash(first) != synth.measurement_hash(other)
    differing = [
        step
        for step in range(8)
        if not np.array_equal(first.frames[step]["cam0"].values, other.frames[step]["cam0"].values)
    ]
    assert differing, "a different sequence seed must change the measured samples"
    for step in range(8):
        assert np.array_equal(first.frames[step]["cam0"].values, second.frames[step]["cam0"].values)


# --------------------------------------------------------------------------------------
# CFA phase and coordinates (aligned with owac.raw.rawvla_sequence)
# --------------------------------------------------------------------------------------


def test_cfa_plane_indices_match_the_reference_tool_ordering() -> None:
    """The local tile/plane mapping reproduces the RawVLA plane coordinates."""
    for pattern in sorted(reference.CFA_PATTERNS):
        tiles = synth.cfa_plane_tile_indices(pattern)
        expected = reference.cfa_plane_positions(pattern, (0, 0))
        for plane, tile in enumerate(tiles):
            assert (float(tile // 2) + 0.5, float(tile % 2) + 0.5) == (
                float(expected[plane][0]),
                float(expected[plane][1]),
            )


def test_cfa_pixel_plane_index_follows_the_pattern_phase() -> None:
    for pattern in sorted(reference.CFA_PATTERNS):
        index = synth.cfa_pixel_plane_index(pattern, 4, 4)
        for row in range(4):
            for col in range(4):
                colour = reference.CFA_PATTERNS[pattern][2 * (row % 2) + (col % 2)]
                plane = int(index[row, col])
                plane_colour = reference.PLANE_COLOURS[plane]
                assert plane_colour[0] == colour
                plane_row, plane_col = divmod(synth.cfa_plane_tile_indices(pattern)[plane], 2)
                assert (row % 2, col % 2) == (plane_row, plane_col)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("offset", [(0, 0), (1, 1), (1, 0), (0, 1), (2, 2), (3, 1), (2, 4)])
def test_cfa_crop_keeps_phase_coordinates_and_samples(pattern: str, offset: tuple) -> None:
    """A crop re-declares the window phase and keeps every true sample coordinate."""
    rows, cols = 8, 8
    mosaic = np.arange(rows * cols, dtype=np.uint16).reshape(rows, cols)
    measurement = synth.SensorMeasurement(
        camera_id="c",
        input_mode="bayer_cfa",
        values=mosaic,
        unit="raw_dn_uint16_mosaic",
        dtype_name="uint16",
        valid_mask=np.ones((rows, cols), dtype=bool),
        bad_mask=np.zeros((rows, cols), dtype=bool),
        missing_mask=np.zeros((rows, cols), dtype=bool),
        saturated_mask=np.zeros((rows, cols), dtype=bool),
        saturation_tail_probability=np.zeros((rows, cols), dtype=np.float32),
        frame_index=0,
        crop_offset=(0, 0),
        cfa_pattern=pattern,
        plane_positions=reference.cfa_plane_positions(pattern, (0, 0)),
        metadata={
            "black_level": 0.0,
            "white_level": 4095.0,
            "bit_depth": 12,
            "gain": 1.0,
            "exposure_ms": 1.0,
        },
    )
    row0, col0 = offset
    crop = synth.crop_sensor_measurement(measurement, row0=row0, col0=col0, height=4, width=4)
    assert crop.crop_offset == (row0, col0)
    assert crop.cfa_pattern == synth.rotate_cfa_pattern(pattern, row0 % 2, col0 % 2)
    if row0 % 2 == 0 and col0 % 2 == 0:
        assert crop.cfa_pattern == pattern

    planes = crop.plane_set()
    reference_planes = reference.cfa_plane_values(crop.values, crop.cfa_pattern, (row0, col0))
    assert np.array_equal(planes.values, reference_planes.values)
    assert np.array_equal(
        planes.positions, reference.cfa_plane_positions(crop.cfa_pattern, (row0, col0))
    )

    coordinates = planes.coordinates()
    rows_index = np.rint(coordinates[..., 0] - 0.5).astype(int)
    cols_index = np.rint(coordinates[..., 1] - 0.5).astype(int)
    assert (rows_index >= row0).all() and (rows_index < row0 + 4).all()
    assert (cols_index >= col0).all() and (cols_index < col0 + 4).all()
    assert np.array_equal(planes.values, mosaic[rows_index, cols_index])
    letters = reference.CFA_PATTERNS[pattern]
    for plane in range(4):
        expected_letter = reference.PLANE_COLOURS[plane][0]
        sampled = [
            letters[2 * (int(row) % 2) + (int(col) % 2)]
            for row, col in zip(rows_index[plane].ravel(), cols_index[plane].ravel(), strict=True)
        ]
        assert set(sampled) == {expected_letter}


def test_cfa_crop_recounts_masks_and_rejects_bad_sizes() -> None:
    measurement = synth.SensorMeasurement(
        camera_id="c",
        input_mode="bayer_cfa",
        values=np.zeros((8, 8), dtype=np.uint16),
        unit="raw_dn_uint16_mosaic",
        dtype_name="uint16",
        valid_mask=np.ones((8, 8), dtype=bool),
        bad_mask=np.zeros((8, 8), dtype=bool),
        missing_mask=np.zeros((8, 8), dtype=bool),
        saturated_mask=np.zeros((8, 8), dtype=bool),
        saturation_tail_probability=np.zeros((8, 8), dtype=np.float32),
        frame_index=0,
        crop_offset=(0, 0),
        cfa_pattern="RGGB",
        plane_positions=reference.cfa_plane_positions("RGGB", (0, 0)),
        metadata={
            "black_level": 0.0,
            "white_level": 4095.0,
            "bit_depth": 12,
            "gain": 1.0,
            "exposure_ms": 1.0,
        },
    )
    with pytest.raises(ValueError):
        synth.crop_sensor_measurement(measurement, row0=0, col0=0, height=3, width=4)
    with pytest.raises(ValueError):
        synth.crop_sensor_measurement(measurement, row0=6, col0=0, height=4, width=4)


# --------------------------------------------------------------------------------------
# the two declared observation models
# --------------------------------------------------------------------------------------


def test_cfa_and_raw3_are_separate_declared_models() -> None:
    """The mosaic model and the co-sited pseudo-RAW model differ in every declaration."""
    scene = background_module_scene(background=0.5)
    cfa = synth.MeasurementModel(kind="cfa", response=(0.9, 1.0, 1.0, 0.7))
    raw3 = synth.MeasurementModel(kind="raw3", response=(0.9, 1.0, 0.7))
    cfa_sequence = synth.generate_sequence(
        scene, (cfa_sensor(),), cfa, quiet_schedule(), frames=3, seed=1
    )
    raw_sequence = synth.generate_sequence(
        scene, (raw3_sensor(),), raw3, quiet_schedule(), frames=3, seed=1
    )
    cfa_measurement = cfa_sequence.frames[0]["cam0"]
    raw_measurement = raw_sequence.frames[0]["cam3"]
    assert cfa_measurement.values.dtype == np.uint16
    assert cfa_measurement.values.shape == (48, 64)
    assert raw_measurement.values.dtype == np.float32
    assert raw_measurement.values.shape == (3, 48, 64)
    assert cfa_measurement.unit != raw_measurement.unit
    assert cfa_measurement.cfa_pattern == "RGGB" and raw_measurement.cfa_pattern == ""
    assert cfa_measurement.plane_positions is not None
    assert raw_measurement.plane_positions is None
    assert cfa_measurement.metadata["quantisation"] == "round_to_integer_dn"
    assert raw_measurement.metadata["quantisation"] == "none_float32_storage"
    assert cfa.version != raw3.version
    assert synth.measurement_hash(cfa_sequence) != synth.measurement_hash(raw_sequence)

    with pytest.raises(ValueError):
        synth.generate_sequence(scene, (raw3_sensor(),), cfa, quiet_schedule(), frames=3, seed=1)
    with pytest.raises(ValueError):
        synth.generate_sequence(scene, (cfa_sensor(),), raw3, quiet_schedule(), frames=3, seed=1)


def test_declared_noise_variance_matches_a_measured_flat_field() -> None:
    """The declared shot+read variance is reproduced on a flat field."""
    scene = background_module_scene(background=0.4)
    sensor = cfa_sensor()
    for shot, read in ((400.0, 3.0), (2000.0, 0.0), (600.0, 8.0)):
        model = synth.MeasurementModel(
            kind="cfa", shot_noise_scale=shot, read_noise_e=read, response=(1.0, 1.0, 1.0, 1.0)
        )
        sequence = synth.generate_sequence(
            scene, (sensor,), model, quiet_schedule(4, 3), frames=4, seed=5
        )
        measurement = sequence.frames[0]["cam0"]
        signal = (measurement.values.astype(np.float64) - 64.0) / (4095.0 - 64.0)
        predicted = 0.4 / shot + (read / shot) ** 2
        assert int(measurement.saturated_mask.sum()) == 0
        assert abs(float(signal.mean()) - 0.4) < 0.02
        assert abs(float(signal.var()) / predicted - 1.0) < 0.05
        assert measurement.metadata["read_noise_e"] == pytest.approx(read)
        assert measurement.metadata["shot_noise_scale"] == pytest.approx(shot)


def test_sensor_declared_read_noise_overrides_the_model_default() -> None:
    scene = background_module_scene(background=0.4)
    sensor = cfa_sensor(read_noise_e=5.0)
    sequence = synth.generate_sequence(
        scene, (sensor,), synth.MeasurementModel(kind="cfa"), quiet_schedule(3, 4), frames=3, seed=6
    )
    metadata = sequence.frames[0]["cam0"].metadata
    assert metadata["read_noise_e"] == pytest.approx(5.0)
    assert metadata["read_noise_e_source"] == "sensor"


def test_noise_event_replaces_the_declared_noise_from_its_step() -> None:
    scene = background_module_scene(background=0.4)
    schedule = synth.EventSchedule(
        horizon=6,
        seed=7,
        noise=(synth.NoiseEvent(step=3, shot_noise_scale=120.0, read_noise_e=4.0),),
    )
    sequence = synth.generate_sequence(
        scene, (cfa_sensor(),), synth.MeasurementModel(kind="cfa"), schedule, frames=6, seed=8
    )
    before = sequence.frames[2]["cam0"].metadata
    after = sequence.frames[3]["cam0"].metadata
    assert before["shot_noise_scale"] == pytest.approx(1000.0)
    assert before["read_noise_e_source"] == "model_default"
    assert after["shot_noise_scale"] == pytest.approx(120.0)
    assert after["read_noise_e"] == pytest.approx(4.0)
    assert after["read_noise_e_source"] == "noise_event"


def test_saturation_is_clipped_and_flagged_with_a_tail_statement() -> None:
    """Clipped samples are flagged, and the declared tail predicts their frequency."""
    shape = synth.ShapeSpec(
        shape_id="bright",
        kind="disc",
        scale_px=8.0,
        position_px=(12.0, 12.0),
        velocity_px_per_step=(0.0, 0.0),
        colour_response=(1.0, 1.0, 1.0),
    )
    scene = synth.SceneSpec(
        scene_id="saturating",
        seed=2,
        partition="train",
        shapes=(shape,),
        background_level=0.05,
        camera_motion_px=(0.0, 0.0),
    )
    model = synth.MeasurementModel(
        kind="cfa", shot_noise_scale=1000.0, response=(1.0, 1.0, 1.0, 1.0)
    )
    sequence = synth.generate_sequence(
        scene, (cfa_sensor(),), model, quiet_schedule(3, 9), frames=3, seed=10
    )
    measurement = sequence.frames[0]["cam0"]
    saturation_dn = measurement.metadata["saturation_dn"]
    tail = np.asarray(measurement.saturation_tail_probability, dtype=np.float64)
    saturated = np.asarray(measurement.saturated_mask)
    assert float(measurement.values.max()) <= saturation_dn + 1.0
    assert saturated.any()
    assert (measurement.values[saturated] >= saturation_dn - 1).all()
    assert measurement.metadata["saturation_semantics"].startswith("clipped at saturation_dn")

    rows = np.arange(measurement.values.shape[0], dtype=np.float64)[:, None] - 12.0
    cols = np.arange(measurement.values.shape[1], dtype=np.float64)[None, :] - 12.0
    distance = np.sqrt(rows**2 + cols**2)
    bright_region = distance <= 0.9 * 8.0
    dark_region = distance >= 10.0
    assert bright_region.any() and dark_region.any()
    assert float(tail[bright_region].min()) > 0.3
    assert float(tail[dark_region].max()) < 1e-6
    assert int(saturated[dark_region].sum()) == 0

    expected = float(tail[bright_region].mean())
    observed = float(saturated[bright_region].mean())
    sigma = math.sqrt(max(expected * (1.0 - expected), 1e-12) / int(bright_region.sum()))
    assert abs(observed - expected) < 4.0 * sigma + 0.01
    assert 0.3 < observed < 0.7


def test_raw3_marks_missing_and_bad_samples_with_nan() -> None:
    """The float32 model stores NaN for absent samples; masks stay authoritative."""
    scene = background_module_scene(background=0.5)
    model = synth.MeasurementModel(
        kind="raw3", response=(0.9, 1.0, 0.7), bad_pixel_fraction=0.02, missing_sample_fraction=0.02
    )
    sequence = synth.generate_sequence(
        scene, (raw3_sensor(),), model, quiet_schedule(3, 11), frames=3, seed=12
    )
    measurement = sequence.frames[0]["cam3"]
    absent = measurement.missing_mask | measurement.bad_mask
    assert absent.any()
    assert np.isnan(measurement.values[absent]).all()
    assert np.isfinite(measurement.values[~absent]).all()
    contributing = measurement.contributing_mask()
    assert np.array_equal(contributing, ~absent)
    assert measurement.contributing_count() == int((~absent).sum())
    assert measurement.contributing_count() < measurement.values.size
    assert (measurement.values[~absent] >= 0.0).all()
    assert (measurement.values[~absent] <= measurement.metadata["saturation_dn"]).all()
    assert measurement.metadata["missing_value_semantics"].startswith("stored NaN")


def test_cfa_calibrated_matches_the_reference_normalisation() -> None:
    """``calibrated()`` delegates to the RawVLA black/white normalisation."""
    scene = background_module_scene(background=0.5)
    model = synth.MeasurementModel(
        kind="cfa", bad_pixel_fraction=0.01, missing_sample_fraction=0.01
    )
    sequence = synth.generate_sequence(
        scene, (cfa_sensor(),), model, quiet_schedule(3, 13), frames=3, seed=14
    )
    measurement = sequence.frames[0]["cam0"]
    values, metadata = measurement.calibrated()
    calibration = reference.CfaCalibration(
        black_level=measurement.metadata["black_level"],
        white_level=measurement.metadata["white_level"],
        missing_mask=measurement.missing_mask,
        bad_mask=measurement.bad_mask,
        saturated_mask=measurement.saturated_mask,
    )
    expected, expected_metadata = reference.apply_cfa_calibration(measurement.values, calibration)
    assert np.array_equal(values, expected, equal_nan=True)
    assert metadata["missing_count"] == expected_metadata["missing_count"]
    assert metadata["bad_count"] == expected_metadata["bad_count"]
    absent = measurement.missing_mask | measurement.bad_mask
    assert np.isnan(values[absent]).all()


# --------------------------------------------------------------------------------------
# saturation / quantisation probability statements
# --------------------------------------------------------------------------------------


def test_normal_tail_helper_matches_the_declared_accuracy() -> None:
    """The vectorised tail agrees with ``math.erfc`` inside its declared 1e-7."""
    means = np.linspace(-1.0, 3.0, 401)
    variance = 0.25
    computed = synth.saturation_tail_probability(means, np.full_like(means, variance), 1.0)
    z = (1.0 - means) / math.sqrt(variance)
    expected = np.asarray([0.5 * math.erfc(value / math.sqrt(2.0)) for value in z])
    assert np.max(np.abs(computed - expected)) < 1e-6
    assert ((computed >= 0.0) & (computed <= 1.0)).all()


def test_saturation_tail_is_monotone_and_degenerates_without_variance() -> None:
    means = np.linspace(0.0, 1.5, 31)
    variance = np.full_like(means, 0.01)
    tail = synth.saturation_tail_probability(means, variance, 1.0)
    assert np.all(np.diff(tail) >= -1e-12)
    assert tail[0] < 1e-6 and tail[-1] > 0.9
    assert synth.saturation_tail_probability(np.asarray(1.0), np.asarray(0.0), 1.0) == 1.0
    assert synth.saturation_tail_probability(np.asarray(0.5), np.asarray(0.0), 1.0) == 0.0


def test_quantisation_interval_probabilities_sum_to_one() -> None:
    mean, variance = 0.5, 0.0025
    levels = np.arange(-4000, 8000, 1, dtype=np.float64)
    probabilities = synth.quantisation_interval_probability(levels, mean, variance, step=1.0)
    assert probabilities.min() >= 0.0
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6
    mode = int(np.argmax(probabilities))
    assert abs(float(levels[mode]) - round(mean)) <= 1.0
    with pytest.raises(ValueError):
        synth.quantisation_interval_probability(levels, mean, variance, step=0.0)


# --------------------------------------------------------------------------------------
# multi-camera declarations and cost profile
# --------------------------------------------------------------------------------------


def test_multiple_cameras_with_different_shapes_and_patterns_render_one_frame() -> None:
    scene = synth.generate_scene("test", "multi-0001", canvas=(64, 64), horizon=4)
    sensors = (
        synth.SensorSpec(
            camera_id="left",
            height=48,
            width=64,
            input_mode="bayer_cfa",
            cfa_pattern="GRBG",
            black_level=64.0,
            white_level=4095.0,
            bit_depth=12,
        ),
        synth.SensorSpec(
            camera_id="right",
            height=32,
            width=32,
            input_mode="bayer_cfa",
            cfa_pattern="BGGR",
            black_level=0.0,
            white_level=1023.0,
            bit_depth=10,
            read_noise_e=1.0,
        ),
    )
    model = synth.MeasurementModel(kind="cfa")
    schedule = synth.build_event_schedule(
        scene, frames=4, seed=3, camera_id="left", camera_shift_events=1
    )
    sequence = synth.generate_sequence(scene, sensors, model, schedule, frames=4, seed=15)
    frame = sequence.frames[0]
    assert set(frame) == {"left", "right"}
    assert frame["left"].values.shape == (48, 64)
    assert frame["right"].values.shape == (32, 32)
    assert frame["right"].cfa_pattern == "BGGR"
    assert sequence.inference[0].bundle.camera_ids == ("left", "right")
    for camera_id in ("left", "right"):
        truth_geometry = sequence.truth[0].geometry
        assert f"{camera_id}:image_px" in truth_geometry
        assert truth_geometry[f"{camera_id}:image_px"].shape == (len(scene.shapes), 2)
        assert camera_id in sequence.truth[0].occlusion_mask


def test_measure_cost_reports_real_timings_and_sizes() -> None:
    """Both declared models are actually run; the reported numbers are measured."""
    cfa_cost = synth.measure_cost(
        frames=12, seed=1, model_kind="cfa", cameras=2, height=48, width=64
    )
    raw3_cost = synth.measure_cost(
        frames=12, seed=1, model_kind="raw3", cameras=1, height=48, width=64
    )
    for cost, measurements_per_frame in ((cfa_cost, 2), (raw3_cost, 1)):
        assert cost["frames_per_sequence"] == 12
        assert cost["measured_frames"] == 12
        assert cost["measurements"] == 12 * measurements_per_frame
        assert cost["wall_seconds_total"] > 0.0
        assert cost["wall_ms_per_frame"] > 0.0
        assert cost["wall_seconds_per_sequence"] > 0.0
        assert cost["wall_ms_per_measurement"] > 0.0
        assert cost["tracemalloc_peak_bytes"] > 0
        assert cost["sequence_array_bytes"] > 0
        assert cost["array_bytes_per_frame"] > 0.0
        assert cost["environment"]["numpy"] == np.__version__
        assert cost["p0_scale_claim"] == "small_sample_only"
    assert cfa_cost["model_kind"] == "cfa" and raw3_cost["model_kind"] == "raw3"
    assert cfa_cost["sequence_array_bytes"] != raw3_cost["sequence_array_bytes"]

    repeated = synth.measure_cost(
        frames=12, seed=1, model_kind="cfa", cameras=2, height=48, width=64
    )
    assert repeated["sequence_array_bytes"] == cfa_cost["sequence_array_bytes"]
    assert repeated["measurements"] == cfa_cost["measurements"]
    with pytest.raises(ValueError):
        synth.measure_cost(model_kind="sensor")


#: Declared P0 small-sample probe: 8 sequences x 32 frames, once per declared model.
P0_PROBE_SEQUENCES = 8
P0_PROBE_FRAMES = 32
P0_PROBE_CAMERAS = {"cfa": 2, "raw3": 1}
P0_PROBE_ARTIFACT = "outputs/n02-full/p0/synthetic/sample_cost.json"


def repository_root() -> Path:
    """Repo root of this checkout, found from the test file itself."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise AssertionError("repository root not found")


def test_p0_sample_cost_artifact_records_a_real_small_sample_run() -> None:
    """Run the declared P0 probe for both models and write the real cost artifact."""
    costs = {
        kind: synth.measure_cost(
            frames=P0_PROBE_FRAMES,
            seed=917,
            model_kind=kind,
            cameras=cameras,
            height=64,
            width=64,
            sequences=P0_PROBE_SEQUENCES,
        )
        for kind, cameras in P0_PROBE_CAMERAS.items()
    }
    for kind, cameras in P0_PROBE_CAMERAS.items():
        cost = costs[kind]
        assert cost["sequences"] == P0_PROBE_SEQUENCES
        assert cost["measured_frames"] == P0_PROBE_SEQUENCES * P0_PROBE_FRAMES
        assert cost["measurements"] == cost["measured_frames"] * cameras
        assert cost["wall_seconds_total"] > 0.0
        assert cost["wall_ms_per_frame"] > 0.0
        assert cost["sequence_array_bytes"] > 0

    artifact = {
        "stage": "p0",
        "generator_version": synth.SYNTHETIC_VERSION,
        "probe": {
            "sequences_per_model": P0_PROBE_SEQUENCES,
            "frames_per_sequence": P0_PROBE_FRAMES,
            "cameras_per_model": dict(P0_PROBE_CAMERAS),
            "frame_shape": [64, 64],
            "seed": 917,
        },
        "cost": costs,
        "declared_targets": {
            "P1_train_frames": synth.P1_TRAIN_FRAME_TARGET,
            "P2_sequences": synth.P2_SEQUENCE_TARGET,
            "P2_sequence_frames": list(synth.P2_SEQUENCE_FRAMES),
            "status": "targets_declared_not_built",
        },
        "p0_scale_claim": "small_sample_only",
    }
    artifact_path = repository_root() / P0_PROBE_ARTIFACT
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written["cost"] == json.loads(json.dumps(costs))
    assert written["generator_version"] == synth.SYNTHETIC_VERSION
    assert written["p0_scale_claim"] == "small_sample_only"
    assert written["declared_targets"]["status"] == "targets_declared_not_built"

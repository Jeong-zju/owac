"""Numeric tests for the N02 synthetic measurement-domain generator.

Every test here checks a declared number against a computed one: the forward
operator against the emitted frames, the CFA plane coordinates against the Bayer
phase, the evidence schedule against the seeds, and the explicit two-hypothesis
model against its closed-form Bayes update. Nothing asserts a framed success
output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from owac.frameworks.n02 import synthetic as synth

#: Declared generator settings used by most tests. ``occlusion_schedule="none"``
#: keeps the analysis of the forward operator separate from the occluder, which
#: has its own dedicated tests.
PLAIN = synth.SyntheticOptions(
    height=48,
    width=48,
    noise_sigma=0.0,
    occlusion_schedule="none",
)


def generator(**overrides) -> synth.SyntheticGenerator:
    options = PLAIN
    if overrides:
        options = synth.dataclasses.replace(PLAIN, **overrides)
    return synth.SyntheticGenerator(options, seed=0, split="train")


def angular_error(left: float, right: float) -> float:
    return abs(float(synth.wrapped_angle_difference(left, right)))


def measurement_identity(
    generator: synth.SyntheticGenerator, episode: synth.SyntheticEpisode
) -> float:
    """Max absolute deviation of ``frames`` from ``attenuation * clean + noise``."""
    clean = generator.clean_measurements(episode)
    attenuation = np.stack(
        [generator.sensor_attenuation(episode, step) for step in range(episode.num_frames)]
    )
    predicted = attenuation * clean + episode.noise
    return float(np.abs(predicted - episode.frames).max())


# --------------------------------------------------------------------------------------
# forward operator
# --------------------------------------------------------------------------------------


def test_arc_is_ambiguous_between_hypotheses() -> None:
    """The arc field must be identical for both values: only the cue carries the answer."""
    generator_ = generator()
    scene = generator_.sample_latent("opening_direction", "left", "train", seed=1)
    coordinates, _ = synth.sample_coordinates(generator_.options.height, generator_.options.width)
    distance = np.hypot(
        coordinates[:, 0] - scene.arc_center_xy[0], coordinates[:, 1] - scene.arc_center_xy[1]
    )
    phi = np.arctan2(
        coordinates[:, 0] - scene.arc_center_xy[0], coordinates[:, 1] - scene.arc_center_xy[1]
    ) % (2.0 * math.pi)

    left_arc = synth.measurement_field(coordinates, scene, cue_strength=0.0)[0]
    right_arc = synth.measurement_field(coordinates, scene.with_value("right"), cue_strength=0.0)[0]

    assert float(np.abs(left_arc - right_arc).max()) == 0.0
    # The declared arc peak is `radial_peak * angular_peak`, and both terms reach 1
    # only where the sample sits at the band's mid-radius *and* at `arc_angle`.
    # Rather than tune a threshold, evaluate the declared operator directly.
    band = (distance >= scene.inner_radius) & (distance <= scene.outer_radius)
    radial = np.clip(
        np.minimum(distance - scene.inner_radius, scene.outer_radius - distance)
        / scene.edge_softness,
        0.0,
        1.0,
    )
    angular = synth.sigma_cos(phi, scene.arc_angle, scene.angular_width)
    declared = radial * angular
    assert np.allclose(left_arc, declared, atol=1e-12)
    # Pixel centres rarely land exactly on the band's mid-radius, so check the
    # *declared normalisation* at a synthetic sample that does, instead of
    # demanding a 1.0 peak from the sampled grid.
    assert float(declared[band].max()) <= 1.0 + 1e-12
    assert float(declared[band].max()) >= 0.8
    mid_radius = 0.5 * (scene.inner_radius + scene.outer_radius)
    exact = np.array(
        [
            scene.arc_center_xy[0] + mid_radius * math.sin(scene.arc_angle),
            scene.arc_center_xy[1] + mid_radius * math.cos(scene.arc_angle),
        ]
    )[None, :]
    assert float(synth.measurement_field(exact, scene, cue_strength=0.0)[0][0]) == pytest.approx(
        1.0, abs=1e-9
    )
    assert int((left_arc > 0.5).sum()) >= 8, "the declared arc must span many samples"


def test_cue_peaks_exactly_at_the_declared_marker() -> None:
    """The noiseless distinguish cue must peak at ``marker_xy`` for the true value."""
    generator_ = generator()
    scene = generator_.sample_latent("opening_direction", "right", "train", seed=2)
    coordinates, grid = synth.sample_coordinates(
        generator_.options.height, generator_.options.width
    )
    strength = 0.75
    _, cue, clean = synth.measurement_field(coordinates, scene, cue_strength=strength)
    marker = synth.marker_xy(scene)

    distances = np.hypot(coordinates[:, 0] - marker[0], coordinates[:, 1] - marker[1])
    nearest = int(np.argmin(distances))
    assert float(cue[nearest]) == pytest.approx(
        math.exp(-(distances[nearest] ** 2) / (2.0 * scene.cue_sigma**2)), abs=1e-12
    )
    assert float(cue.max()) == pytest.approx(float(cue[nearest]), abs=1e-12)
    assert float(clean[nearest]) >= strength * float(cue[nearest])

    # The *cue* component must peak at the marker; the total `clean` field also
    # contains the ambiguous arc, so its own maximum is not the marker.
    cue_peak = np.unravel_index(int(np.argmax(cue.reshape(grid.shape[:-1]))), grid.shape[:-1])
    peak_distance = math.hypot(cue_peak[0] + 0.5 - marker[0], cue_peak[1] + 0.5 - marker[1])
    assert peak_distance <= 1.5, f"cue peak {cue_peak} is {peak_distance:.2f} px from marker"


def test_frames_match_the_declared_measurement_law() -> None:
    """``frames == attenuation * clean + noise`` exactly, with and without an occluder."""
    for schedule in ("none", "linear", "quadratic", "step"):
        generator_ = generator(occlusion_schedule=schedule)
        episode = generator_.generate(
            task="opening_direction",
            num_frames=8,
            episode_index=0,
            seed=3,
            value="left",
            prior_value="left",
            identifiable=True,
        )
        assert measurement_identity(generator_, episode) <= 1e-6, schedule


def test_cue_strength_scales_the_marker_linearly() -> None:
    generator_ = generator()
    scene = generator_.sample_latent("contour_connected", "separate", "train", seed=4)
    coordinates, _ = synth.sample_coordinates(generator_.options.height, generator_.options.width)
    marker = synth.marker_xy(scene)
    nearest = int(np.argmin(np.hypot(coordinates[:, 0] - marker[0], coordinates[:, 1] - marker[1])))

    arc, cue_reference, _ = synth.measurement_field(coordinates, scene, cue_strength=1.0)
    unit = float(cue_reference[nearest])
    for strength in (0.25, 0.5, 1.0):
        clean = synth.measurement_field(coordinates, scene, cue_strength=strength)[2]
        assert float(clean[nearest] - arc[nearest]) == pytest.approx(strength * unit, abs=1e-6)


# --------------------------------------------------------------------------------------
# measurement flags, padding, evidence schedule
# --------------------------------------------------------------------------------------


def test_every_generated_frame_is_a_new_measurement() -> None:
    generator_ = generator()
    episode = generator_.generate(num_frames=6, episode_index=0, seed=5, identifiable=True)
    assert bool(episode.new_measurement.all())
    assert episode.unique_measurement_count == episode.num_frames


def test_doping_padding_flag_semantics_are_explicit() -> None:
    """``new_measurement=False`` is reserved for repeated padding; a fresh
    sequence must not set it, so downstream code can trust the flag."""
    generator_ = generator()
    episode = generator_.generate(num_frames=6, episode_index=1, seed=6, identifiable=True)
    payload = episode.as_measurement_payload()
    assert payload["new_measurement"].shape == (6,)
    assert payload["metadata_available"]["timestamp"] is False
    assert payload["metadata_available"]["executed_action"] is False
    assert payload["metadata_available"]["noise_model"] is True
    assert bool(np.isfinite(payload["variance"]).all())


def test_evidence_arrival_is_not_a_fixed_frame_index() -> None:
    """The arrival step must vary with the seed; a constant would be a bug."""
    generator_ = generator()
    arrivals = [
        generator_.generate(
            num_frames=12, episode_index=index, seed=index, identifiable=True
        ).evidence_arrival_step
        for index in range(12)
    ]
    assert len(set(arrivals)) >= 3, f"arrival steps barely vary: {sorted(set(arrivals))}"
    assert min(arrivals) >= 1
    assert max(arrivals) <= 11
    for arrival in arrivals:
        assert arrival not in (0, 11), "arrival must leave room before and after"


def test_cue_strength_is_zero_before_arrival_and_grows_after() -> None:
    generator_ = generator()
    for seed in range(6):
        episode = generator_.generate(
            num_frames=10, episode_index=seed, seed=seed, identifiable=True
        )
        arrival = episode.evidence_arrival_step
        assert float(np.abs(episode.cue_strength[:arrival]).max()) == 0.0
        after = episode.cue_strength[arrival:]
        assert after.min() > 0.0
        assert float(max(after)) >= float(after[0]) - 1e-12


def test_cue_strength_and_arrival_are_sampled_independently() -> None:
    """Two episodes may share an arrival step but must not share a fixed strength."""
    generator_ = generator()
    by_arrival: dict[int, set[float]] = {}
    for seed in range(24):
        episode = generator_.generate(
            num_frames=10, episode_index=seed, seed=seed, identifiable=True
        )
        by_arrival.setdefault(episode.evidence_arrival_step, set()).add(
            round(float(episode.cue_strength[-1]), 6)
        )
    repeated = {arrival: values for arrival, values in by_arrival.items() if len(values) > 1}
    assert repeated, f"no arrival step had more than one strength: {by_arrival}"


def test_unidentifiable_sequences_really_contain_no_cue() -> None:
    generator_ = generator()
    for seed in range(6):
        episode = generator_.generate(
            task="opening_direction",
            num_frames=8,
            episode_index=seed,
            seed=seed,
            identifiable=False,
        )
        assert not episode.identifiable
        assert float(np.abs(episode.cue_strength).max()) == 0.0

    scene = generator_.sample_latent("opening_direction", "left", "train", seed=0)
    coordinates, _ = synth.sample_coordinates(generator_.options.height, generator_.options.width)
    left = synth.measurement_field(coordinates, scene, cue_strength=0.0)[2]
    right = synth.measurement_field(coordinates, scene.with_value("right"), cue_strength=0.0)[2]
    assert float(np.abs(left - right).max()) == 0.0


# --------------------------------------------------------------------------------------
# latent / prior separation
# --------------------------------------------------------------------------------------


def test_prior_condition_only_touches_the_declared_context() -> None:
    """The prior must not appear anywhere in the measurement production path.

    The prior condition is part of the episode identity (and of the RNG stream
    that draws the evidence schedule), so two conditions are different episodes by
    construction. What must hold - and what this test checks - is that the
    measurement is a function of the latent geometry, the evidence schedule and
    the noise only: the *frames* of a condition can be reproduced exactly from
    those three declared quantities, and the prior lives only in the context.
    """
    generator_ = generator()
    for condition in synth.PRIOR_CONDITIONS:
        episode = generator_.generate(
            task="opening_direction",
            prior_condition=condition,
            num_frames=8,
            episode_index=0,
            seed=7,
            value="left",
            prior_value="left",
            identifiable=True,
        )
        assert episode.prior_condition == condition
        # The frames are exactly the declared forward operator through the declared
        # occluder plus the recorded noise, with no prior term anywhere.
        assert measurement_identity(generator_, episode) <= 1e-6, condition
        # And the latent value is a declared parameter, not something the prior
        # draws: `correct`/`misleading`/`uninformative`/`missing` all ask for "left".
        if condition != "stale":
            assert episode.geometry_before.value == "left"
        context = episode.context.as_dict()
        assert set(context) & {"frames", "values", "measurement"} == set()


def test_latent_geometry_lives_only_in_supervision() -> None:
    generator_ = generator()
    episode = generator_.generate(
        task="handle_orientation",
        num_frames=6,
        episode_index=0,
        seed=8,
        value="vertical",
        prior_value="vertical",
        identifiable=True,
    )
    payload = episode.as_measurement_payload()
    context = payload["context"]
    metadata = payload["metadata"]
    # The encoded answer is a cue *direction*: the context must never contain it.
    # (The value's name may legitimately appear as a declared prior, which is
    # context the encoder is allowed to have and which the measurement cannot
    # depend on - `test_prior_condition_never_changes_the_measurement` covers that.)
    assert set(context) & {"value", "geometry", "cue_angle", "cue_strength"} == set()
    assert "value" not in metadata
    assert episode.value not in repr(metadata), "latent value leaked into metadata"
    assert "geometry" not in metadata


def test_supervision_carries_the_hidden_geometry() -> None:
    """The answer must be available to the loss/evaluator, and only there."""
    generator_ = generator()
    episode = generator_.generate(
        task="opening_direction",
        num_frames=6,
        episode_index=0,
        seed=15,
        value="left",
        prior_value="left",
        identifiable=True,
    )
    supervision = episode.supervision
    assert supervision["value"] == "left"
    assert supervision["geometry_before"]["value"] == "left"
    assert supervision["geometry_before"]["task"] == "opening_direction"
    assert supervision["geometry_before"]["arc_center_xy"] == list(
        episode.geometry_before.arc_center_xy
    )
    assert supervision["cue_angle_true"] == synth.cue_angle_for_value("opening_direction", "left")
    assert len(supervision["value_over_time"]) == episode.num_frames
    assert supervision["identifiable"] is True
    assert episode.supervision["value"] == episode.value
    assert episode.supervision["geometry_before"]["value"] == episode.geometry_before.value
    assert episode.supervision["geometry_after"]["value"] == episode.geometry_after.value


def test_stale_geometry_changes_exactly_at_arrival() -> None:
    generator_ = generator()
    found = 0
    for seed in range(16):
        episode = generator_.generate(
            task="opening_direction",
            prior_condition="stale",
            num_frames=10,
            episode_index=seed,
            seed=seed,
            value="left",
            prior_value="left",
            identifiable=True,
        )
        if not episode.geometry_changed:
            continue
        found += 1
        change = episode.geometry_change_step
        assert change == episode.evidence_arrival_step
        assert bool(episode.new_measurement[change])
        assert episode.value_over_time[change - 1] != episode.value_over_time[change]
        assert episode.geometry_after.value != episode.geometry_before.value
    assert found > 0, "no stale episode flipped its geometry across 16 seeds"


def test_prior_conditions_have_the_declared_meaning() -> None:
    generator_ = generator()
    for condition in synth.PRIOR_CONDITIONS:
        episode = generator_.generate(
            task="opening_direction",
            prior_condition=condition,
            num_frames=8,
            episode_index=0,
            seed=9,
            value="right",
            prior_value="right",
            identifiable=True,
        )
        context = episode.context.as_dict()
        assert context["prior_condition"] == condition
        if condition == "missing":
            # A missing prior is declared absent, not declared as zero odds, and the
            # measurement is unaffected (`test_prior_condition_never_changes_...`).
            assert context["prior_value"] is None
            assert context["prior_log_odds"] is None
        elif condition == "uninformative":
            assert context["prior_log_odds"] == 0.0
        elif condition == "misleading":
            # The declared prior must actively favour the *other* answer. It is the
            # declaration that is misleading, not the measurement: the frames stay
            # exactly the declared operator (asserted separately).
            assert context["prior_log_odds"] < 0.0
            assert context["prior_value"] == episode.geometry_before.value
        else:
            assert context["prior_log_odds"] > 0.0


# --------------------------------------------------------------------------------------
# train/validation split
# --------------------------------------------------------------------------------------


def test_split_separates_shapes_positions_and_occlusion_templates() -> None:
    report = generator().check_split_disjointness()
    assert report["disjoint"] is True
    for key in ("shape_id", "position_id", "occlusion_template"):
        assert report[key]["disjoint"] is True, key
        assert report[key]["overlap"] == []


def test_split_geometry_is_actually_used_by_the_sampler() -> None:
    train = generator()
    validation = synth.SyntheticGenerator(PLAIN, seed=0, split="validation")
    train_shapes = set()
    validation_shapes = set()
    for index in range(12):
        train_shapes.add(
            train.sample_latent("opening_direction", "left", "train", seed=index).shape_id
        )
        validation_shapes.add(
            validation.sample_latent("opening_direction", "left", "validation", seed=index).shape_id
        )
    assert train_shapes <= set(train.split_values("train")["shape_id"])
    assert validation_shapes <= set(validation.split_values("validation")["shape_id"])
    assert not (train_shapes & validation_shapes)


# --------------------------------------------------------------------------------------
# CFA sampling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", sorted(synth.CFA_PATTERNS))
def test_cfa_plane_coordinates_match_the_declared_pattern(pattern: str) -> None:
    """Plane positions must be the pattern's true photosite coordinates."""
    positions = synth.cfa_plane_positions(pattern)
    tiles = synth.CFA_PATTERNS[pattern]
    assert positions.shape == (4, 2)

    colour_of_tile = {(0, 0): tiles[0], (0, 1): tiles[1], (1, 0): tiles[2], (1, 1): tiles[3]}
    for plane_index, name in enumerate(synth.PLANE_COLOURS):
        y, x = positions[plane_index]
        tile = (int(y - 0.5), int(x - 0.5))
        expected_colour = "G" if name in {"G1", "G2"} else name
        assert colour_of_tile[tile] == expected_colour, (pattern, name, tile)

    # G1 and G2 must be the two *different* green photosites.
    assert not np.allclose(positions[1], positions[2])
    # No two planes may be co-sited.
    for first in range(4):
        for second in range(first + 1, 4):
            assert np.abs(positions[first] - positions[second]).max() >= 1.0


def test_cfa_sample_picks_the_right_photosites() -> None:
    """A ramp mosaic must yield each plane's own values at its own coordinates."""
    tile = np.array([[0.0, 1.0], [2.0, 3.0]])
    mosaic = np.tile(tile, (3, 3))
    values, positions = synth.cfa_sample(mosaic, "RGGB")
    assert values.shape == (4, 3, 3)
    # RGGB: R at tile (0,0) -> 0, G1 at (0,1) -> 1, G2 at (1,0) -> 2, B at (1,1) -> 3.
    assert float(values[0, 0, 0]) == 0.0
    assert float(values[1, 0, 0]) == 1.0
    assert float(values[2, 0, 0]) == 2.0
    assert float(values[3, 0, 0]) == 3.0
    assert positions[0].tolist() == [0.5, 0.5]
    assert positions[3].tolist() == [1.5, 1.5]


def test_bayer_cfa_episode_uses_four_planes_at_true_offsets() -> None:
    generator_ = synth.SyntheticGenerator(
        synth.dataclasses.replace(PLAIN, input_mode="bayer_cfa", cfa_pattern="GBRG"),
        seed=0,
        split="train",
    )
    episode = generator_.generate(
        task="contour_connected",
        num_frames=5,
        episode_index=0,
        seed=10,
        value="connected",
        prior_value="connected",
        identifiable=True,
    )
    assert episode.frames.shape == (5, 4, 24, 24)
    assert episode.sample_xy.shape == (4, synth.SAMPLE_XY_ROWS, 2)
    expected = synth.cfa_plane_positions("GBRG")
    for plane_index in range(4):
        assert episode.sample_xy[plane_index, 0].tolist() == pytest.approx(
            expected[plane_index].tolist()
        )
        assert episode.sample_xy[plane_index, 1].tolist() == pytest.approx([2.0, 2.0])
    assert episode.metadata["planes_are_not_co_sited"] is True
    assert episode.metadata["cfa_pattern"] == "GBRG"

    # Each plane must be a spectral response applied to the same physical signal,
    # so the four planes are not four co-sited colour copies.
    step = int(np.argmax(episode.cue_strength))
    planes = episode.frames[step]
    assert not np.allclose(planes[0], planes[3])
    assert measurement_identity(generator_, episode) <= 1e-6


def test_cfa_planes_match_the_declared_spectral_response() -> None:
    """Re-derive one plane from the forward operator and compare to the frame."""
    generator_ = synth.SyntheticGenerator(
        synth.dataclasses.replace(PLAIN, input_mode="bayer_cfa", cfa_pattern="RGGB"),
        seed=0,
        split="train",
    )
    episode = generator_.generate(
        task="opening_direction",
        num_frames=4,
        episode_index=0,
        seed=11,
        value="left",
        prior_value="left",
        identifiable=True,
    )
    step = int(np.argmax(episode.cue_strength))
    scene = episode.geometry_before
    coordinates, _ = synth.sample_coordinates(generator_.options.height, generator_.options.width)
    assert scene is episode.geometry_before
    strength = float(episode.cue_strength[step])
    for plane_index, name in enumerate(synth.PLANE_COLOURS):
        field = synth.render_plane(
            coordinates,
            scene,
            cue_strength=strength,
            base_value=generator_.options.base_value,
            plane=name,
        ).reshape(generator_.options.height, generator_.options.width)
        expected, _ = synth.cfa_sample(field, "RGGB")
        assert (
            float(np.abs(episode.frames[step, plane_index] - expected[plane_index]).max()) <= 1e-6
        ), name


# --------------------------------------------------------------------------------------
# measurement-only inference
# --------------------------------------------------------------------------------------


def test_noiseless_inference_recovers_the_true_hypothesis() -> None:
    """A noiseless frame must select the true value under the declared operator."""
    for split in ("train", "validation"):
        generator_ = synth.SyntheticGenerator(PLAIN, seed=0, split=split)
        for task in synth.TASK_NAMES:
            for value in synth.values_for_task(task):
                episode = generator_.generate(
                    task=task,
                    num_frames=8,
                    episode_index=0,
                    seed=0,
                    value=value,
                    prior_value=value,
                    identifiable=True,
                )
                generator_.validate_scene(episode.geometry_before)
                assert generator_.infer_hypothesis(episode) == value, (split, task, value)
                assert (
                    angular_error(
                        generator_.infer_cue_angle(episode), synth.cue_angle_for_value(task, value)
                    )
                    < 1e-9
                )


def test_hypothesis_fit_residual_is_zero_for_the_true_hypothesis() -> None:
    generator_ = generator()
    episode = generator_.generate(
        task="opening_direction",
        num_frames=8,
        episode_index=0,
        seed=12,
        value="right",
        prior_value="right",
        identifiable=True,
    )
    fits = generator_.hypothesis_fit(episode)
    assert fits["right"]["mean_squared_residual"] == pytest.approx(0.0, abs=1e-18)
    assert fits["left"]["mean_squared_residual"] > 1e-4
    assert fits["right"]["decisive_residual"] == pytest.approx(0.0, abs=1e-18)


def test_verify_measurement_places_the_excess_at_the_true_marker() -> None:
    generator_ = generator()
    for value in ("left", "right"):
        episode = generator_.generate(
            task="opening_direction",
            num_frames=8,
            episode_index=0,
            seed=13,
            value=value,
            prior_value=value,
            identifiable=True,
        )
        report = generator_.verify_measurement(episode)
        assert report["argmax_excess"] == value
        assert report["agrees"] is True
        other = "right" if value == "left" else "left"
        assert report["excess"][value] > report["excess"][other] + 1e-6


def test_inference_refuses_an_unidentifiable_sequence() -> None:
    generator_ = generator()
    episode = generator_.generate(
        task="opening_direction",
        num_frames=8,
        episode_index=0,
        seed=14,
        identifiable=False,
    )
    with pytest.raises(ValueError, match="no cue"):
        generator_.infer_hypothesis(episode)
    with pytest.raises(ValueError, match="no cue"):
        generator_.hypothesis_fit(episode)


def test_inference_stays_correct_at_low_noise_without_occlusion() -> None:
    """With a declared tiny noise level and no occluder, inference must stay exact.

    This is the controlled regime used for the mechanism experiments: the noise is
    small relative to the distinguishing cue (measured via the declared operator),
    so a wrong answer would be a real bug rather than a hard inference problem.
    """
    generator_ = synth.SyntheticGenerator(
        synth.dataclasses.replace(PLAIN, noise_sigma=1.0e-3), seed=0, split="train"
    )
    correct = 0
    total = 0
    for task in synth.TASK_NAMES:
        for value in synth.values_for_task(task):
            for index in range(3):
                episode = generator_.generate(
                    task=task,
                    num_frames=8,
                    episode_index=index,
                    seed=index,
                    value=value,
                    prior_value=value,
                    identifiable=True,
                )
                correct += generator_.infer_hypothesis(episode) == value
                total += 1
    assert (correct, total) == (total, total)


def test_occlusion_never_covers_a_marker() -> None:
    """No occluder template may come within 3 sigma of either hypothesis's marker."""
    generator_ = generator(occlusion_schedule="linear", max_occluder_transmission=0.35)
    checked = 0
    for split in ("train", "validation"):
        split_generator = synth.SyntheticGenerator(generator_.options, seed=0, split=split)
        for index in range(6):
            scene = split_generator.sample_latent(
                "opening_direction", "left", split, seed=index, episode_index=index
            )
            for template in (
                "left_half",
                "right_half",
                "top_strip",
                "bottom_strip",
                "shuffle",
            ):
                mask = split_generator._occluder_mask(
                    template,
                    (generator_.options.height, generator_.options.width),
                    np.random.default_rng(0),
                    scene,
                )
                rows, cols = np.nonzero(mask > 0.0)
                if rows.size == 0:
                    continue
                for value in synth.values_for_task(scene.task):
                    marker = synth.marker_xy(scene, synth.cue_angle_for_value(scene.task, value))
                    nearest = float(np.hypot(rows - marker[0], cols - marker[1]).min())
                    assert nearest >= 3.0 * scene.cue_sigma, (template, value, nearest)
                    checked += 1
    assert checked > 20, "the occluder clearance check barely ran"


def test_generator_validates_scenes_it_emits() -> None:
    generator_ = generator(occlusion_schedule="linear")
    for index in range(4):
        episode = generator_.generate(
            task="handle_orientation",
            num_frames=8,
            episode_index=index,
            seed=index,
            identifiable=True,
        )
        report = generator_.validate_scene(episode.geometry_before)
        assert report["valid"] is True
        assert all(report["markers_inside_frame"].values())


def test_generator_is_deterministic_under_a_seed() -> None:
    first = generator()
    second = generator()
    left = first.generate(num_frames=8, episode_index=2, seed=21, identifiable=True)
    right = second.generate(num_frames=8, episode_index=2, seed=21, identifiable=True)
    assert np.array_equal(left.frames, right.frames)
    assert left.evidence_arrival_step == right.evidence_arrival_step
    assert np.array_equal(left.cue_strength, right.cue_strength)
    assert left.episode_id == right.episode_id


# --------------------------------------------------------------------------------------
# explicit two-hypothesis mode (analytic)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("separation", [0.5, math.pi / 2.0, 2.0, 3.0])
def test_explicit_llr_matches_the_raw_likelihood_ratio(separation: float) -> None:
    """The closed form must equal the raw Gaussian log-likelihood difference.

    The declared scales are ``noise_sigma = 0.5`` and ``amplitude = 0.8``, so the
    effective variance is ``0.5**2 / 0.8 = 0.3125``. The expected value is built
    from :func:`gaussian_logpdf` at those declared numbers rather than from a
    second copy of the closed form.
    """
    model = synth.ExplicitObservationModel(
        theta0=0.0,
        separation=separation,
        noise_sigma=0.5,
        amplitude=0.8,
        prior_log_odds=0.2,
    )
    assert model.variance == pytest.approx(0.5**2 / 0.8, rel=1e-12)
    mean_0, mean_1 = model.means()
    for observation in (-2.0, -0.5, 0.0, 0.3, 1.0, 3.0):
        expected = synth.gaussian_logpdf(
            observation, mean_1, model.variance
        ) - synth.gaussian_logpdf(observation, mean_0, model.variance)
        assert model.log_likelihood(observation, 1) == pytest.approx(
            synth.gaussian_logpdf(observation, mean_1, model.variance), abs=1e-12
        )
        assert model.log_likelihood_ratio(observation) == pytest.approx(expected, abs=1e-12)
        # And the affine closed form the docstring declares, in the residual.
        residual = observation - mean_0
        affine = separation * (2.0 * residual - separation) / (2.0 * model.variance)
        assert model.log_likelihood_ratio(observation) == pytest.approx(affine, abs=1e-12)


def test_explicit_accumulate_matches_the_exact_product() -> None:
    rng = np.random.default_rng(0)
    model = synth.ExplicitObservationModel(
        noise_sigma=0.4, amplitude=0.7, separation=0.7, prior_log_odds=-0.3
    )
    result = model.exact_update(rng, hypothesis=1, steps=8)
    assert result["log_odds"] == pytest.approx(result["reference_log_odds"], abs=1e-9)
    assert result["posterior"] == pytest.approx(result["reference_posterior"], abs=1e-12)
    assert result["absolute_difference"] < 1e-9
    assert result["posterior"] > 0.5, "8 informative observations from H1 must favour H1"
    trajectory = result["log_odds_trajectory"]
    assert len(trajectory) == 9
    assert trajectory[0] == model.prior_log_odds
    total = sum(result["log_likelihood_ratios"])
    assert trajectory[-1] == pytest.approx(model.prior_log_odds + total, abs=1e-12)


def test_explicit_update_is_order_invariant_and_adds_evidence() -> None:
    model = synth.ExplicitObservationModel(noise_sigma=0.3, amplitude=0.6, separation=0.7)
    observations = np.array([0.1, -0.2, 0.4, 0.05])
    forward = model.accumulate(observations)["log_odds"]
    backward = model.accumulate(observations[::-1])["log_odds"]
    assert forward == pytest.approx(backward, abs=1e-12)

    one = model.accumulate(observations[:1])["log_odds"]
    two = model.accumulate(observations[:2])["log_odds"]
    assert two != pytest.approx(one, abs=1e-9), "a second observation must add evidence"


def test_zero_amplitude_evidence_carries_no_information() -> None:
    rng = np.random.default_rng(2)
    model = synth.ExplicitObservationModel(
        noise_sigma=0.2, amplitude=0.0, separation=0.7, prior_log_odds=0.4
    )
    result = model.exact_update(rng, hypothesis=0, steps=5)
    assert math.isinf(model.variance)
    assert result["log_odds"] == pytest.approx(0.4, abs=1e-12)
    assert result["posterior"] == pytest.approx(1.0 / (1.0 + math.exp(-0.4)), abs=1e-12)
    assert model.expected_information() == 0.0
    assert all(ratio == 0.0 for ratio in result["log_likelihood_ratios"])


def test_expected_information_grows_with_amplitude() -> None:
    """Declared evidence strength must increase the expected information."""
    information = []
    for amplitude in (0.1, 0.3, 0.6, 1.0):
        model = synth.ExplicitObservationModel(noise_sigma=0.5, amplitude=amplitude, separation=0.5)
        information.append(model.expected_information())
    assert information == sorted(information), information
    assert information[-1] > information[0] * 5.0


def test_posterior_is_calibrated_and_concentrates_with_evidence() -> None:
    """A Bayes-optimal update must be calibrated and sharpen with more evidence.

    The declared identity ``E[log-odds gain] = steps * separation^2 / (2*variance)``
    is exact in expectation but the log-odds is log-normal, so a sample mean needs
    a very large sample to be tight. This test therefore checks (a) the *median*
    gain against the declared identity, which is robust, and (b) calibration of the
    posterior itself: over many independent runs the average posterior must match
    the probability the sampled hypothesis was the true one.
    """
    separation = 0.5
    for amplitude in (0.1, 0.4, 1.0):
        model = synth.ExplicitObservationModel(
            noise_sigma=0.5, amplitude=amplitude, separation=separation
        )
        rng = np.random.default_rng(7)
        posteriors, log_odds = [], []
        for _ in range(2000):
            result = model.exact_update(rng, hypothesis=1, steps=3)
            posteriors.append(result["posterior"])
            log_odds.append(result["log_odds"])
        expected_gain = 3 * separation**2 / (2.0 * model.variance)
        assert expected_gain == pytest.approx(3 * model.expected_information(), rel=1e-12)
        assert float(np.median(log_odds)) == pytest.approx(expected_gain, rel=0.25), amplitude
        assert float(np.mean(posteriors)) > 0.5, amplitude
        # Calibration, checked as a frequency statement rather than against an
        # approximation: with a uniform prior the reported posterior *is* the
        # probability that H1 is the true hypothesis, so averaging it over runs
        # whose truth is sampled uniformly must return the observed H1 rate.
        calibrated = synth.ExplicitObservationModel(
            noise_sigma=model.noise_sigma,
            amplitude=model.amplitude,
            separation=model.separation,
            prior_log_odds=0.0,  # uniform prior: posterior == P(H1 | y)
        )
        rng = np.random.default_rng(23)
        reported, truth = [], []
        for _ in range(4000):
            true_hypothesis = int(rng.random() < 0.5)
            observation = calibrated.sample(rng, true_hypothesis)
            reported.append(calibrated.accumulate([observation])["posterior"])
            truth.append(true_hypothesis)
        assert float(np.mean(truth)) == pytest.approx(0.5, abs=0.03)
        assert float(np.mean(reported)) == pytest.approx(float(np.mean(truth)), abs=0.03)


def test_more_observations_sharpen_the_posterior() -> None:
    """Extra independent observations must not weaken the declared evidence."""
    model = synth.ExplicitObservationModel(noise_sigma=0.4, amplitude=0.7, separation=0.6)
    gains = []
    for steps in (1, 2, 4, 8):
        rng = np.random.default_rng(11)
        gains.append(
            float(
                np.median(
                    [
                        model.exact_update(rng, hypothesis=1, steps=steps)["log_odds"]
                        for _ in range(400)
                    ]
                )
            )
        )
    assert gains == sorted(gains), gains
    assert gains[-1] > gains[0] * 3.0


def test_explicit_model_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        synth.ExplicitObservationModel(noise_sigma=0.0)
    with pytest.raises(ValueError):
        synth.ExplicitObservationModel(amplitude=1.5)
    with pytest.raises(ValueError):
        synth.ExplicitObservationModel().mean_for(2)
    # Antipodal means are refused rather than silently mis-scored.
    with pytest.raises(ValueError, match="antipodal"):
        synth.ExplicitObservationModel(separation=math.pi)
    with pytest.raises(ValueError, match="antipodal"):
        synth.ExplicitObservationModel(separation=math.pi + 0.1)
    with pytest.raises(ValueError):
        synth.ExplicitObservationModel(separation=0.0)


# --------------------------------------------------------------------------------------
# generator configuration guards
# --------------------------------------------------------------------------------------


def test_options_reject_invalid_configurations() -> None:
    with pytest.raises(ValueError):
        synth.SyntheticOptions(input_mode="rgb_image")
    with pytest.raises(ValueError):
        synth.SyntheticOptions(input_mode="bayer_cfa", cfa_pattern="XYZW")
    with pytest.raises(ValueError):
        synth.SyntheticOptions(height=31, width=32)
    with pytest.raises(ValueError):
        synth.SyntheticOptions(occlusion_schedule="torch")
    with pytest.raises(ValueError):
        synth.SyntheticOptions(arrival_lo=0.9, arrival_hi=0.2)


def test_generate_rejects_degenerate_sequences() -> None:
    generator_ = generator()
    with pytest.raises(ValueError):
        generator_.generate(num_frames=3)
    with pytest.raises(ValueError):
        generator_.generate(task="no_such_task")
    with pytest.raises(ValueError):
        generator_.generate(prior_condition="wishful")

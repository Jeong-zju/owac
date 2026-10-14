"""Scene, schedule and dataset tests for the N02 full synthetic generator.

Every test compares a declared property against a computed one: the derived seed
against its inputs, the partition vocabularies against each other, the dataset
counters against the sequences actually yielded, and the event schedule against
its own declared class times. Nothing here asserts a framed success output.
"""

from __future__ import annotations

import numpy as np
import pytest

from owac.frameworks.n02_full import synthetic as synth

CANVAS = (64, 64)
FRAMES = 12


def schedule_for(scene: synth.SceneSpec, *, frames: int = FRAMES, seed: int = 11, **overrides):
    return synth.build_event_schedule(
        scene, frames=frames, seed=seed, camera_id="cam0", **overrides
    )


def flat_sensors() -> tuple[synth.SensorSpec, ...]:
    return (
        synth.SensorSpec(
            camera_id="cam0",
            height=48,
            width=64,
            input_mode="bayer_cfa",
            cfa_pattern="RGGB",
            black_level=64.0,
            white_level=4095.0,
            bit_depth=12,
        ),
    )


# --------------------------------------------------------------------------------------
# seeds and scene identity
# --------------------------------------------------------------------------------------


def test_scene_seed_is_derived_from_partition_and_scene_id() -> None:
    """Same partition+scene_id reproduces the seed; the id changes it."""
    first = synth.scene_seed("train", "scene-0001")
    assert first == synth.scene_seed("train", "scene-0001")
    assert first != synth.scene_seed("train", "scene-0002")
    assert first != synth.scene_seed("test", "scene-0001")

    derived = [
        synth.scene_seed(partition, f"scene-{index:04d}")
        for partition in ("train", "test")
        for index in range(8)
    ]
    assert len(set(derived)) == len(derived)
    assert not set(derived) & set(synth.MODEL_SEEDS)


def test_scene_hash_is_reproducible_and_input_sensitive() -> None:
    """The scene hash covers the layout, the declared plans and the seed."""
    scene = synth.generate_scene("train", "hash-0001", canvas=CANVAS, horizon=FRAMES)
    repeated = synth.generate_scene("train", "hash-0001", canvas=CANVAS, horizon=FRAMES)
    assert synth.scene_hash(scene) == synth.scene_hash(repeated)
    assert scene.seed == repeated.seed

    other_id = synth.generate_scene("train", "hash-0002", canvas=CANVAS, horizon=FRAMES)
    assert synth.scene_hash(other_id) != synth.scene_hash(scene)

    explicit = synth.generate_scene("train", "hash-0001", seed=5, canvas=CANVAS, horizon=FRAMES)
    assert synth.scene_hash(explicit) != synth.scene_hash(scene)


def test_generated_scenes_stay_inside_the_declared_canvas() -> None:
    """Declared layout rule: every shape centre lies inside the margin band."""
    margin = synth.VOCABULARIES["train"].order_margin_px
    for partition in ("train", "dev", "calibration", "test"):
        vocabulary = synth.VOCABULARIES[partition]
        for index in range(6):
            scene = synth.generate_scene(
                partition, f"{partition}-{index:04d}", canvas=CANVAS, horizon=FRAMES
            )
            assert 1 <= len(scene.shapes) <= 12
            assert len(scene.shapes) in vocabulary.object_counts
            for shape in scene.shapes:
                assert shape.kind in vocabulary.shape_kinds
                scale = shape.scale_px
                y, x = shape.position_px
                assert margin + scale <= y <= CANVAS[0] - margin - scale
                assert margin + scale <= x <= CANVAS[1] - margin - scale
                assert 0.0 <= min(shape.colour_response)
                assert max(shape.colour_response) <= 1.0
                assert np.asarray(shape.velocity_px_per_step).shape == (2,)


def test_scene_diversity_varies_shapes_scale_position_and_background() -> None:
    """The mechanism set must vary the declared axes, not one fixed template."""
    scenes = [
        synth.generate_scene("train", f"diverse-{index:04d}", canvas=CANVAS, horizon=FRAMES)
        for index in range(16)
    ]
    kinds = {kind for scene in scenes for kind in scene.shape_kinds}
    backgrounds = {round(scene.background_level, 6) for scene in scenes}
    scales = {round(shape.scale_px, 6) for scene in scenes for shape in scene.shapes}
    positions = {
        tuple(np.round(shape.position_px, 6)) for scene in scenes for shape in scene.shapes
    }
    assert len(kinds) >= 2
    assert len(backgrounds) >= 8
    assert len(scales) >= 8
    assert len(positions) >= 8


# --------------------------------------------------------------------------------------
# partition vocabularies
# --------------------------------------------------------------------------------------


def test_train_and_sealed_test_vocabularies_are_disjoint() -> None:
    """Train and the sealed test set may not share any declared design axis."""
    train = synth.VOCABULARIES["train"]
    test = synth.VOCABULARIES["test"]
    assert not set(train.shape_kinds) & set(test.shape_kinds)
    assert not set(train.object_counts) & set(test.object_counts)
    assert not set(train.occlusion_modes) & set(test.occlusion_modes)
    assert not set(train.interaction_structures) & set(test.interaction_structures)

    observed_kinds = {"train": set(), "test": set()}
    observed_counts = {"train": set(), "test": set()}
    for partition in ("train", "test"):
        for index in range(8):
            scene = synth.generate_scene(
                partition, f"{partition}-{index:04d}", canvas=CANVAS, horizon=FRAMES
            )
            observed_kinds[partition] |= set(scene.shape_kinds)
            observed_counts[partition].add(len(scene.shapes))
    assert not observed_kinds["train"] & observed_kinds["test"]
    assert not observed_counts["train"] & observed_counts["test"]


def test_scene_shape_combinations_do_not_overlap_across_partitions() -> None:
    """Shape sets of the two families are disjoint, so P1 has a sealed test set."""
    train_sets = {
        tuple(sorted(synth.generate_scene("train", f"combo-{index:04d}").shape_kinds))
        for index in range(8)
    }
    test_sets = {
        tuple(sorted(synth.generate_scene("test", f"combo-{index:04d}").shape_kinds))
        for index in range(8)
    }
    assert train_sets and test_sets
    assert all(set(combo) <= set(synth.VOCABULARIES["train"].shape_kinds) for combo in train_sets)
    assert all(set(combo) <= set(synth.VOCABULARIES["test"].shape_kinds) for combo in test_sets)
    assert not train_sets & test_sets


# --------------------------------------------------------------------------------------
# event schedule
# --------------------------------------------------------------------------------------


def test_event_schedule_is_reproducible_and_covers_every_declared_type() -> None:
    """Same seed reproduces the program; the defaults realise every event class."""
    scene = synth.generate_scene("train", "events-0001", canvas=CANVAS, horizon=FRAMES)
    first = schedule_for(scene)
    second = schedule_for(scene)
    assert synth.schedule_hash(first) == synth.schedule_hash(second)
    assert first.schedule_hash() == synth.schedule_hash(first)

    assert set(first.event_types()) == set(synth.EVENT_TYPES)
    for name in synth.EVENT_TYPES:
        event_type = synth.EVENT_FIELDS[name]
        events = getattr(first, event_type)
        assert events, name
        assert len(set(events)) == len(events), f"{name} repeated an identical event"
        for event in events:
            assert 1 <= int(event.step) < FRAMES, name
            for attribute in ("shape_id", "occluder_id", "occluded_id"):
                value = getattr(event, attribute, None)
                if value is not None:
                    assert value in scene.shape_ids

    different = schedule_for(scene, seed=12)
    assert synth.schedule_hash(different) != synth.schedule_hash(first)


def test_camera_shift_events_stay_within_four_pixels() -> None:
    """The declared displacement class is +-1..4 px and moves at least one axis."""
    scene = synth.generate_scene("test", "shift-0001", canvas=CANVAS, horizon=FRAMES)
    schedule = schedule_for(scene, camera_shift_events=4)
    assert schedule.camera_shift
    for event in schedule.camera_shift:
        assert 1 <= max(abs(event.dy), abs(event.dx)) <= 4
        assert event.camera_id == "cam0"
    with pytest.raises(ValueError):
        synth.CameraShiftEvent(step=1, camera_id="cam0", dy=5, dx=0)


def test_event_classes_draw_independent_times() -> None:
    """Changing one class's count leaves the other classes' times untouched."""
    scene = synth.generate_scene("train", "independence-0001", canvas=CANVAS, horizon=FRAMES)
    base = schedule_for(scene)
    more_exposure = schedule_for(scene, exposure_events=4)
    assert more_exposure.steps("exposure_change") != base.steps("exposure_change")
    for name in ("camera_shift", "object_motion", "occlusion", "relation_change", "identity_swap"):
        assert more_exposure.steps(name) == base.steps(name), name

    fewer_motion = schedule_for(scene, motion_events=1, relation_events=1)
    assert fewer_motion.steps("camera_shift") == base.steps("camera_shift")
    assert fewer_motion.steps("exposure_change") == base.steps("exposure_change")


def test_schedule_rejects_events_outside_the_horizon() -> None:
    with pytest.raises(ValueError):
        synth.EventSchedule(horizon=4, seed=0, exposure=(synth.ExposureEvent(step=9, scale=0.5),))


# --------------------------------------------------------------------------------------
# declared events have real effects
# --------------------------------------------------------------------------------------


def mechanism_scene() -> synth.SceneSpec:
    shapes = (
        synth.ShapeSpec("back", "disc", 8.0, (24.0, 24.0), (0.0, 0.0), (1.0, 1.0, 1.0)),
        synth.ShapeSpec("front", "square", 10.0, (24.0, 60.0), (0.0, 0.0), (0.2, 0.2, 0.2)),
    )
    return synth.SceneSpec(
        scene_id="mechanism",
        seed=3,
        partition="train",
        shapes=shapes,
        background_level=0.2,
        camera_motion_px=(0.0, 0.0),
        occlusion_plan=(synth.OcclusionPlan("front", "back", 2, 10, "sweep_x"),),
    )


def mechanism_sequence(schedule: synth.EventSchedule) -> synth.SyntheticSequence:
    sensor = synth.SensorSpec(
        camera_id="cam0",
        height=48,
        width=48,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
        black_level=64.0,
        white_level=4095.0,
        bit_depth=12,
    )
    return synth.generate_sequence(
        mechanism_scene(),
        (sensor,),
        synth.MeasurementModel(kind="cfa"),
        schedule,
        frames=min(12, schedule.horizon),
        seed=2,
    )


def test_occlusion_event_makes_the_occluded_object_geometrically_hidden() -> None:
    """The scripted sweep must really hide the occluded object inside its window."""
    schedule = synth.EventSchedule(
        horizon=12,
        seed=5,
        occlusion=(
            synth.OcclusionEvent(2, "front", "back", True),
            synth.OcclusionEvent(10, "front", "back", False),
        ),
    )
    sequence = mechanism_sequence(schedule)
    fractions = [float(sequence.truth[step].visible_fraction["cam0"][0]) for step in range(12)]
    masks = [bool(sequence.truth[step].occlusion_mask["cam0"][0]) for step in range(12)]
    assert fractions[0] == 1.0 and fractions[1] == 1.0
    assert any(masks)
    hidden = int(np.argmax(np.asarray(masks)))
    assert fractions[hidden] == 0.0
    assert min(fractions[2:10]) == 0.0
    labels = {label for step in range(12) for label in sequence.truth[step].event_labels}
    assert any(label.startswith("occlusion@2") for label in labels)
    assert any(label.startswith("occlusion_active") for label in labels)


def test_identity_swap_moves_appearance_but_not_object_identity_or_geometry() -> None:
    schedule = synth.EventSchedule(
        horizon=8, seed=6, identity_swap=(synth.IdentitySwapEvent(3, "back", "front"),)
    )
    sequence = mechanism_sequence(schedule)
    assert dict(sequence.truth[2].identity_links) == {"back": "back", "front": "front"}
    assert dict(sequence.truth[3].identity_links) == {"back": "front", "front": "back"}
    assert sequence.truth[3].object_ids == ("back", "front")
    assert np.array_equal(
        sequence.truth[2].geometry["world_px"], sequence.truth[3].geometry["world_px"]
    )
    assert (
        sequence.frames[2]["cam0"].values.tobytes() != sequence.frames[3]["cam0"].values.tobytes()
    )


def test_false_split_adds_a_declared_artefact_without_a_truth_object() -> None:
    schedule = synth.EventSchedule(
        horizon=8,
        seed=7,
        false_split=(synth.AppearanceEvent(2, "back", "", (3.0, 3.0), 4),),
    )
    sequence = mechanism_sequence(schedule)
    assert len(sequence.truth[3].object_ids) == 2
    assert any(label.startswith("false_split_active") for label in sequence.truth[3].event_labels)
    assert not any(label.startswith("false_split") for label in sequence.truth[0].event_labels)
    # the artefact is rendered: the frame inside the window differs from the previous one
    assert (
        sequence.frames[1]["cam0"].values.tobytes() != sequence.frames[2]["cam0"].values.tobytes()
    )


# --------------------------------------------------------------------------------------
# dataset iterator and counters
# --------------------------------------------------------------------------------------


def test_dataset_counters_separate_independent_scenes_from_augmented_views() -> None:
    """Augmented views are counted apart and never inflate the scene count."""
    dataset = synth.generate_dataset(
        sensors=flat_sensors(),
        measurement_model=synth.MeasurementModel(kind="cfa"),
        partitions=("train", "test"),
        scenes=3,
        frames=6,
        views_per_scene=2,
        seed=3,
        canvas=(48, 48),
        event_counts={"occlusion_events": 1, "false_merge_events": 0, "false_split_events": 0},
    )
    sequences = list(dataset)
    counters = dataset.counters
    assert len(sequences) == 2 * 3 * 2
    assert counters.independent_scenes == 2 * 3
    assert counters.augmented_views == 2 * 3
    assert counters.sequences == 2 * 3 * 2
    assert counters.frames == sum(sequence.num_frames for sequence in sequences)
    assert counters.frames == 6 * 12
    assert counters.measurements == counters.frames * len(flat_sensors())
    assert counters.events == sum(
        sum(sequence.manifest["event_counts"].values()) for sequence in sequences
    )

    independent = [s for s in sequences if s.manifest["view_kind"] == "independent"]
    augmented = [s for s in sequences if s.manifest["view_kind"] == "augmented_exposure"]
    assert len(independent) == counters.independent_scenes
    assert len(augmented) == counters.augmented_views
    assert {s.manifest["scene_hash"] for s in independent} == {
        s.manifest["scene_hash"] for s in augmented
    }
    for view in augmented:
        exposure = view.manifest["event_counts"]["exposure_change"]
        assert exposure >= 1

    manifest = dataset.manifest()
    assert manifest["counters"] == counters.as_dict()
    assert manifest["augmentation"]["counted_as"] == "augmented_views"
    assert manifest["p0_scale_claim"] == "small_sample_only"
    assert manifest["declared_targets"]["status"] == "targets_declared_not_built"


def test_dataset_reports_declared_targets_without_claiming_them() -> None:
    dataset = synth.generate_dataset(
        sensors=flat_sensors(),
        measurement_model=synth.MeasurementModel(kind="cfa"),
        partitions=("train",),
        scenes=1,
        frames=4,
        seed=1,
        canvas=(48, 48),
        event_counts={"occlusion_events": 0, "false_merge_events": 0, "false_split_events": 0},
    )
    sequence = next(iter(dataset))
    declared = sequence.manifest["declared_targets"]
    assert declared["P1_train_frames"] == synth.P1_TRAIN_FRAME_TARGET
    assert declared["P2_sequences"] == synth.P2_SEQUENCE_TARGET
    assert declared["P2_sequence_frames"] == list(synth.P2_SEQUENCE_FRAMES)
    assert declared["status"] == "targets_declared_not_built"
    assert dataset.counters.frames == sequence.num_frames


def test_frame_target_planner_maps_the_declared_scale_to_dataset_parameters() -> None:
    """The P1/P2 targets map to generator parameters without claiming they ran."""
    assert synth.sequences_for_frame_target(
        synth.P1_TRAIN_FRAME_TARGET, frames_per_sequence=64
    ) == (synth.P1_TRAIN_FRAME_TARGET // 64)
    assert synth.sequences_for_frame_target(65, frames_per_sequence=32) == 3
    assert synth.sequences_for_frame_target(synth.P2_SEQUENCE_TARGET, frames_per_sequence=64) == (
        -(-synth.P2_SEQUENCE_TARGET // 64)
    )
    with pytest.raises(ValueError):
        synth.sequences_for_frame_target(0, frames_per_sequence=16)
    with pytest.raises(ValueError):
        synth.sequences_for_frame_target(1000, frames_per_sequence=0)


def test_dataset_produces_different_scenes_for_different_scene_ids() -> None:
    dataset = synth.generate_dataset(
        sensors=flat_sensors(),
        measurement_model=synth.MeasurementModel(kind="cfa"),
        partitions=("dev",),
        scenes=5,
        frames=5,
        seed=2,
        canvas=(64, 64),
        event_counts={"occlusion_events": 0, "false_merge_events": 0, "false_split_events": 0},
    )
    sequences = list(dataset)
    hashes = {sequence.manifest["scene_hash"] for sequence in sequences}
    assert len(hashes) == len(sequences) == 5
    kinds = {kind for sequence in sequences for kind in sequence.manifest["shape_kinds"]}
    assert kinds <= set(synth.VOCABULARIES["dev"].shape_kinds)


def test_scene_generation_rejects_unknown_partition_and_accepts_small_canvas() -> None:
    with pytest.raises(ValueError):
        synth.generate_scene("holdout", "x")
    small = synth.generate_scene("test", "tiny-0001", canvas=(16, 16), horizon=6)
    for shape in small.shapes:
        assert 0.0 <= shape.position_px[0] <= 16.0
        assert 0.0 <= shape.position_px[1] <= 16.0

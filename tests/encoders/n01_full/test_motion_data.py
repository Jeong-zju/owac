"""Behavioural properties of the synthetic RAW motion pairs.

The oracles are independent NumPy re-implementations of the *declared* scene
semantics (the similarity pose, the sinusoid radiance field, the exposure
quadrature) plus structural walks over the returned objects.  Nothing here calls
the generator to check the generator: the analytic displacement of a sub-pixel
token is recomputed from the token itself, and the field values are recomputed
from the published geometry and field definition.
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.backends.raw_correspondence import RawSensorFrame
from owac.frameworks.n01_full.config import CFA_PATTERNS
from owac.raw.cfa import CFA_PATTERNS as CFA_PATTERN_COLORS
from owac.raw.n01_motion_data import (
    CASE_KINDS,
    CFA_MATERIAL_CLASS,
    MOTION_SCHEMA,
    PSEUDO_RAW_MATERIAL_CLASS,
    REGIONS,
    SUBPIXEL_TIERS,
    VISIBILITY_KINDS,
    BlobField,
    FrameGeometry,
    Layer,
    MotionDataGenerator,
    P2MotionNoiseConfig,
    SceneFrame,
    SceneRecipe,
    SimilarityPose,
    SinusoidField,
    load_p2_motion_noise_config,
    packed_pull_targets,
    pair_layers,
    parse_subpixel_token,
    render_raw_frame,
    solve_pull,
    subpixel_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/encoders/n01_full/p2_motion_noise.json"
COLOR_INDEX = {"R": 0, "G": 1, "B": 2}


@pytest.fixture(scope="module")
def generator() -> MotionDataGenerator:
    return MotionDataGenerator.from_config_path(CONFIG_PATH)


def small_geometry(**overrides) -> FrameGeometry:
    payload = {"height": 16, "width": 16}
    payload.update(overrides)
    return FrameGeometry(**payload)


# ---------------------------------------------------------------------------
# Analytic displacement truth
# ---------------------------------------------------------------------------


def test_subpixel_tokens_cover_all_tiers_axes_and_signs():
    tokens = subpixel_tokens()
    assert len(tokens) == 2 * len(SUBPIXEL_TIERS) * 2
    tiers = sorted(
        {abs(parse_subpixel_token(token)[0] or parse_subpixel_token(token)[1]) for token in tokens}
    )
    assert tiers == sorted(SUBPIXEL_TIERS)
    for token in tokens:
        dx, dy = parse_subpixel_token(token)
        assert (dx != 0.0) ^ (dy != 0.0)
        assert abs(dx) in SUBPIXEL_TIERS or abs(dy) in SUBPIXEL_TIERS
    positive = [
        token
        for token in tokens
        if parse_subpixel_token(token)[0] > 0 or parse_subpixel_token(token)[1] > 0
    ]
    negative = [
        token
        for token in tokens
        if parse_subpixel_token(token)[0] < 0 or parse_subpixel_token(token)[1] < 0
    ]
    assert len(positive) == len(negative) == len(tokens) // 2


def test_subpixel_truth_is_exactly_the_declared_displacement(generator):
    """The generated truth must equal the token's displacement, sign included."""
    for token in subpixel_tokens():
        pair = generator.sample_pair(split="train", case="translation_subpixel", sequence=token)
        truth = pair.supervision
        dx, dy = parse_subpixel_token(token)
        oracle = truth.sensor_xy_reference + torch.tensor([dx, dy], dtype=torch.float64)
        assert torch.equal(truth.sensor_xy_source, oracle)
        # The displacement is sub-pixel but never zero, and its magnitude is the tier.
        displacement = (truth.sensor_xy_source - truth.sensor_xy_reference).abs().max().item()
        assert displacement == pytest.approx(max(abs(dx), abs(dy)))
        assert 0.0 < displacement < 1.0


def test_subpixel_truth_holds_per_phase_in_packed_units(generator):
    token = "x+0.25"
    pair = generator.sample_pair(split="train", case="translation_subpixel", sequence=token)
    truth = pair.supervision
    geometry = generator.geometry()
    packed = truth.packed_xy_source
    assert tuple(packed.shape) == (4, 32, 32, 2)
    rows, cols = geometry.packed_shape
    for channel in range(4):
        for row, col in ((0, 0), (5, 7), (rows - 1, cols - 1)):
            sensor_y, sensor_x = geometry.sample_sensor_yx(channel, row, col)
            expected_x, expected_y = truth.sensor_xy_source[sensor_y, sensor_x].tolist()
            dy, dx = geometry.channel_offsets[channel]
            oracle = (
                (expected_x - (geometry.origin_yx[1] + dx)) / 2.0,
                (expected_y - (geometry.origin_yx[0] + dy)) / 2.0,
            )
            assert packed[channel, row, col].tolist() == pytest.approx(oracle, abs=1e-12)
        # One packed step spans two sensor pixels: the packed pull minus the query
        # index must be exactly the sensor displacement halved, in every phase.
        rows_index = torch.arange(32, dtype=torch.float64).reshape(32, 1).expand(32, 32)
        cols_index = torch.arange(32, dtype=torch.float64).reshape(1, 32).expand(32, 32)
        query = torch.stack([cols_index, rows_index], dim=-1)
        expected = torch.tensor([0.25 / 2.0, 0.0], dtype=torch.float64)
        assert torch.allclose(packed[channel] - query, expected, atol=1e-12)


def test_packed_targets_use_the_source_frame_origin(generator):
    source_geometry = FrameGeometry(height=64, width=64, origin_yx=(2, 2))
    pair = generator.sample_pair(
        split="train",
        case="translation_subpixel",
        sequence="x+0.5",
        source_geometry=source_geometry,
    )
    assert pair.visible.source.sensor_origin_xy == (2, 2)
    truth = pair.supervision
    for channel in range(4):
        dy, dx = source_geometry.channel_offsets[channel]
        sensor = truth.sensor_xy_source
        oracle = torch.stack(
            [
                (sensor[..., 0] - (source_geometry.origin_yx[1] + dx)) / 2.0,
                (sensor[..., 1] - (source_geometry.origin_yx[0] + dy)) / 2.0,
            ],
            dim=-1,
        )
        rows, cols = pair.visible.reference.values.shape[1:]
        row_index = 2 * torch.arange(rows).reshape(-1, 1) + dy
        col_index = 2 * torch.arange(cols).reshape(1, -1) + dx
        assert torch.allclose(truth.packed_xy_source[channel], oracle[row_index, col_index])


def test_channel_offsets_follow_the_phase_permutation():
    """A non-trivial permutation produces the declared, possibly negative offsets."""
    geometry = FrameGeometry(height=16, width=16, origin_yx=(1, 0), channel_phase=(2, 3, 0, 1))
    assert geometry.channel_offsets == ((0, 0), (0, 1), (-1, 0), (-1, 1))
    assert geometry.grid_origin_yx == (0, 0)
    assert geometry.sample_sensor_yx(2, 0, 0) == (0, 0)
    assert geometry.sample_sensor_yx(0, 0, 0) == (1, 0)
    with pytest.raises(ValueError, match="parity"):
        FrameGeometry(height=16, width=16, origin_yx=(1, 1), channel_phase=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="permutation"):
        FrameGeometry(height=16, width=16, channel_phase=(0, 1, 2, 2))


def test_rotation_truth_matches_an_independent_rotation(generator):
    """The shipped rotation case is exactly the declared rotation about the centre."""
    recipe = generator.config.recipe("rotation")
    assert recipe.jitter_px == 0.0
    pair = generator.sample_pair(split="train", case="rotation", sequence="0")
    geometry = generator.geometry()
    centre = np.array([(geometry.width - 1) / 2.0, (geometry.height - 1) / 2.0])
    theta = np.deg2rad(recipe.rotation_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = pair.supervision.sensor_xy_reference.numpy().reshape(-1, 2)
    oracle = (xy - centre) @ rotation.T + centre
    np.testing.assert_allclose(
        pair.supervision.sensor_xy_source.numpy().reshape(-1, 2),
        oracle,
        rtol=0.0,
        atol=1e-12,
    )
    displacement = pair.supervision.sensor_xy_source - pair.supervision.sensor_xy_reference
    assert float(displacement.abs().max()) > 0.1


# ---------------------------------------------------------------------------
# Geometry, occlusion and the physical meaning of a valid correspondence
# ---------------------------------------------------------------------------


def test_valid_correspondence_carries_the_same_radiance():
    """With an even (phase preserving) shift, valid pulls must match sample for sample."""
    geometry = small_geometry()
    shift = (2.0, 0.0)
    source = SceneFrame(
        frame_id=0,
        time=0.0,
        layers=(Layer("background", SinusoidField(period_xy=(8.0, 8.0)), SimilarityPose()),),
    )
    reference = SceneFrame(
        frame_id=1,
        time=1.0,
        layers=(
            Layer(
                "background",
                SinusoidField(period_xy=(8.0, 8.0)),
                SimilarityPose(translation_xy=shift),
            ),
        ),
    )
    solve = solve_pull(pair_layers(reference, source), geometry, geometry)
    reference_values = render_raw_frame(
        reference,
        geometry,
        _calibration(),
        dataset_id="d",
        episode_id="e",
        camera_id="c",
        add_noise=False,
    ).values
    source_values = render_raw_frame(
        source,
        geometry,
        _calibration(),
        dataset_id="d",
        episode_id="e",
        camera_id="c",
        add_noise=False,
    ).values
    assert int(solve.valid.sum()) > 0
    rows, cols = geometry.packed_shape
    checked = 0
    for channel in range(geometry.channels):
        row_index, col_index = geometry.grid_indices_yx(channel)
        channel_valid = solve.valid[row_index.reshape(-1)][:, col_index.reshape(-1)]
        interior = torch.zeros_like(channel_valid)
        interior[:, :-1] = True
        mask = (channel_valid & interior)[:, :-1]
        assert int(mask.sum()) > 0
        # The pull is (x + 2, y): one packed column to the right in the source.
        assert torch.allclose(
            reference_values[channel][:, :-1][mask],
            source_values[channel][:, 1:][mask],
            atol=2e-5,
        )
        checked += int(mask.sum())
    assert checked > 0 and rows * cols > 0


def _calibration():
    """The declared synthetic calibration of the shipped config."""
    return load_p2_motion_noise_config(CONFIG_PATH).noise


def test_layered_occlusion_labels_are_consistent(generator):
    pair = generator.sample_pair(split="train", case="layered_occlusion", sequence="0")
    truth = pair.supervision
    counts = truth.visibility_counts()
    assert counts["valid"] > 0
    assert counts["occluded_in_source"] > 0
    assert counts["out_of_bounds"] > 0
    assert 0.0 < truth.diagnostics["occluder_visible_fraction"] < 1.0
    occluded = truth.visibility == VISIBILITY_KINDS.index("occluded_in_source")
    assert bool(
        (truth.visible_layer_source[occluded] != truth.visible_layer_reference[occluded]).all()
    )
    # The mirror case (a nearer layer in the reference) is structurally impossible
    # with a rigid source layer: the pull is chart-consistent by construction.
    assert not bool((truth.visible_layer_source < truth.visible_layer_reference).any())
    assert bool((truth.valid == (truth.visibility == VISIBILITY_KINDS.index("valid"))).all())
    geometry = generator.geometry()
    origin_y, origin_x = geometry.grid_origin_yx
    height, width = geometry.sensor_shape
    out = truth.visibility == VISIBILITY_KINDS.index("out_of_bounds")
    pull = truth.sensor_xy_source
    outside = (
        (pull[..., 0] < origin_x - 1e-9)
        | (pull[..., 0] > origin_x + width - 1 + 1e-9)
        | (pull[..., 1] < origin_y - 1e-9)
        | (pull[..., 1] > origin_y + height - 1 + 1e-9)
    )
    assert bool((out == outside)[out].all())


def test_local_nonrigid_case_deforms_only_the_current_frame(generator):
    frames = generator.frames(split="train", case="local_nonrigid", sequence="0", length=3)
    assert frames[0].layers[0].warp is None
    for frame in frames[1:]:
        assert frame.layers[0].warp is not None
        assert frame.layers[0].warp.amplitude == pytest.approx(1.5)
    pair = generator.sample_pair(split="train", case="local_nonrigid", sequence="0")
    # The deformation is centred on the sensor centre and vanishes at the corners.
    truth = pair.supervision
    displacement = (truth.sensor_xy_source - truth.sensor_xy_reference).abs()
    centre = (truth.sensor_xy_reference.shape[0] // 2, truth.sensor_xy_reference.shape[1] // 2)
    assert float(displacement[centre].max()) > float(displacement[0, 0].max()) + 0.5
    assert float(displacement[centre].max()) > 1.0
    assert float(displacement[0, 0].max()) < 1.0


# ---------------------------------------------------------------------------
# Determinism, causality and the seed rule
# ---------------------------------------------------------------------------


def test_same_namespace_is_bit_identical_and_regenerated_independently(generator):
    from_config = MotionDataGenerator.from_config_path(CONFIG_PATH)
    first = generator.sample_pair(split="train", case="combined_hard", sequence="7")
    second = from_config.sample_pair(split="train", case="combined_hard", sequence="7")
    assert torch.equal(first.visible.reference.values, second.visible.reference.values)
    assert torch.equal(first.visible.source.values, second.visible.source.values)
    assert torch.equal(first.supervision.sensor_xy_source, second.supervision.sensor_xy_source)
    other = generator.sample_pair(split="train", case="combined_hard", sequence="8")
    assert not torch.equal(first.visible.reference.values, other.visible.reference.values)


def test_seed_rule_matches_an_independent_hash_and_stays_separate():
    rule = load_p2_motion_noise_config(CONFIG_PATH).seed_rule
    namespace = rule.namespace("train", "motion_blur", "3")
    assert namespace == "n01-full-v1/train/motion_blur/3"
    oracle = (
        int.from_bytes(
            hashlib.sha256(f"{rule.data_seed_base}|{namespace}".encode()).digest()[:8], "big"
        )
        % rule.modulus
    )
    assert rule.seed(namespace) == oracle
    seeds = {
        rule.seed(rule.namespace(split, case, sequence))
        for split in REGIONS
        for case in ("motion_blur", "rotation")
        for sequence in ("0", "1")
    }
    assert len(seeds) == 16
    assert seeds.isdisjoint(set(rule.model_seeds))
    assert rule.data_seed_base not in rule.model_seeds
    assert rule.frame_seed(namespace, "noise", 0) != rule.frame_seed(namespace, "noise", 1)
    with pytest.raises(ValueError, match="separate"):
        dataclasses.replace(rule, data_seed_base=rule.model_seeds[0])
    with pytest.raises(ValueError, match="region"):
        rule.namespace("test-set", "motion_blur", "0")
    with pytest.raises(ValueError, match="without"):
        rule.namespace("train", "motion_blur", "0/1")


def test_no_future_frame_leakage(generator):
    """Truncating a sequence must leave every earlier frame bit-identical."""
    long_frames = generator.frames(split="train", case="combined_hard", sequence="1", length=5)
    short_frames = generator.frames(split="train", case="combined_hard", sequence="1", length=2)
    assert len(long_frames) == 5
    assert (len(long_frames), len(short_frames)) == (5, 2)
    for index in range(2):
        assert long_frames[index] == short_frames[index]
    rendered_long = [
        generator.render(frame, split="train", case="combined_hard", sequence="1")
        for frame in long_frames[:2]
    ]
    rendered_short = [
        generator.render(frame, split="train", case="combined_hard", sequence="1")
        for frame in short_frames
    ]
    for left, right in zip(rendered_long, rendered_short, strict=True):
        assert torch.equal(left.values, right.values)
    pairs = generator.sample_sequence(split="train", case="combined_hard", sequence="1", length=5)
    assert len(pairs) == 4
    for pair in pairs:
        truth = pair.supervision
        assert truth.source_frame_id < truth.reference_frame_id
        assert truth.source_time < truth.reference_time
        assert pair.visible.source.frame_id < pair.visible.reference.frame_id
        assert pair.visible.source.time < pair.visible.reference.time
    blocks = pairs[-1].visible.metadata["noise_block_id"]
    assert str(pairs[-1].visible.reference.frame_id) in blocks["reference"]


# ---------------------------------------------------------------------------
# The model-visible half carries no truth
# ---------------------------------------------------------------------------


FORBIDDEN_FIELD_NAMES = {
    "supervision",
    "sensor_xy_source",
    "sensor_xy_reference",
    "packed_xy_source",
    "visibility",
    "visible_layer_reference",
    "visible_layer_source",
    "layer_ids",
    "blur_length_px",
    "blur_samples",
    "exposure_gain_reference",
    "exposure_gain_source",
    "ambiguity_period_px",
    "diagnostics",
}


def walk(value, path="$"):
    """Yield ``(path, field name, value)`` for every node reachable from ``value``."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        yield path, path.rsplit(".", 1)[-1], value
        for item in dataclasses.fields(value):
            yield from walk(getattr(value, item.name), f"{path}.{item.name}")
    elif isinstance(value, dict | type({}.items())):
        for name, item in value.items():
            yield from walk(item, f"{path}.{name}")
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")
    else:
        yield path, path.rsplit(".", 1)[-1], value


def test_model_visible_half_contains_no_geometric_truth(generator):
    truth_names = []
    tensors = []
    for path, name, value in walk(
        generator.sample_pair(split="train", case="layered_occlusion", sequence="0").visible
    ):
        assert name not in FORBIDDEN_FIELD_NAMES, f"{path} carries a truth field"
        truth_names.append(path)
        if isinstance(value, torch.Tensor):
            tensors.append(value)
    assert any(".values" in name for name in truth_names)
    assert tensors, "the walk must reach the RAW tensors, otherwise it proves nothing"
    assert hasattr(RawSensorFrame, "__dataclass_fields__")
    for field_name in RawSensorFrame.__dataclass_fields__:
        assert field_name not in FORBIDDEN_FIELD_NAMES
    pair = generator.sample_pair(split="train", case="layered_occlusion", sequence="0")
    truth = pair.supervision
    for tensor in (
        truth.sensor_xy_source,
        truth.valid,
        truth.visibility,
        truth.packed_xy_source,
    ):
        assert all(tensor is not candidate for candidate in tensors)
    assert all(value is not truth for _, _, value in walk(pair.visible))


def test_visible_metadata_is_the_declared_vocabulary(generator):
    metadata = generator.sample_pair(
        split="train", case="exposure_gain", sequence="0"
    ).visible.metadata
    assert set(metadata) == {
        "split",
        "case",
        "sequence",
        "material_class",
        "input_mode",
        "is_real_cfa",
        "value_units",
        "calibration_version",
        "packed_grid",
        "exposure",
        "noise_block_id",
        "timestamps",
    }
    assert metadata["exposure"]["known"] is True
    assert metadata["exposure"]["reference_gain"] == pytest.approx(1890.0)
    assert metadata["exposure"]["source_gain"] == pytest.approx(1400.0)
    unknown = generator.sample_pair(
        split="train", case="combined_hard", sequence="0"
    ).visible.metadata
    assert unknown["exposure"]["known"] is False
    assert unknown["exposure"]["reference_gain"] is None
    assert unknown["exposure"]["source_gain"] is None
    # The unknown is recorded as unknown, and the true value stays in the supervision.
    assert generator.sample_pair(
        split="train", case="combined_hard", sequence="0"
    ).supervision.exposure_gain_reference == pytest.approx(1890.0)
    assert metadata["packed_grid"]["packed_step_sensor_px"] == 2
    assert metadata["packed_grid"]["channel_offsets_yx"] == [[0, 0], [0, 1], [1, 0], [1, 1]]


# ---------------------------------------------------------------------------
# CFA versus pseudo-RAW
# ---------------------------------------------------------------------------


def test_cfa_and_pseudo_raw_entries_stay_distinct(generator):
    cfa = generator.sample_pair(split="train", case="rotation", sequence="0")
    frame = cfa.visible.reference
    geometry = generator.geometry()
    assert frame.values.shape == (4, geometry.height // 2, geometry.width // 2)
    assert frame.channel_phase == (0, 1, 2, 3)
    assert frame.cfa_pattern in CFA_PATTERNS
    assert frame.validity is not None and frame.validity.shape == frame.values.shape
    assert cfa.visible.metadata["material_class"] == CFA_MATERIAL_CLASS
    assert cfa.visible.metadata["is_real_cfa"] is False
    assert frame.calibration_version == generator.calibration.calibration_version

    rgb3 = generator.sample_pair(
        split="train", case="rotation", sequence="0", input_mode="rawvla_rgb3"
    )
    pseudo = rgb3.visible.reference
    assert pseudo.values.shape == (3, geometry.height, geometry.width)
    assert pseudo.channel_phase == ()
    assert pseudo.cfa_pattern == "rgb3"
    assert rgb3.visible.metadata["material_class"] == PSEUDO_RAW_MATERIAL_CLASS
    assert rgb3.visible.metadata["material_class"] != cfa.visible.metadata["material_class"]
    assert "no CFA phase" in rgb3.visible.metadata["packed_grid"]["note"]
    assert rgb3.supervision.packed_xy_source.shape[:1] == torch.Size([3])
    assert pseudo.values.shape[0] != 4


def test_rendered_samples_match_an_independent_field_evaluation():
    """Recompute the field, the CFA colour selection and the exposure in NumPy."""
    for geometry in (
        small_geometry(),
        small_geometry(input_mode="rawvla_rgb3", cfa_pattern="rgb3", channel_phase=()),
    ):
        field = SinusoidField(period_xy=(9.0, 7.0))
        scene = SceneFrame(
            frame_id=3,
            time=2.5,
            layers=(Layer("background", field, SimilarityPose(translation_xy=(1.5, -0.5))),),
            exposure_gain=1400.0,
        )
        calibration = _calibration()
        rendered = render_raw_frame(
            scene,
            geometry,
            calibration,
            dataset_id="d",
            episode_id="e",
            camera_id="c",
            add_noise=False,
        )
        assert rendered.frame_id == 3
        assert rendered.time == 2.5
        period = np.array([9.0, 7.0])
        phases = np.array([0.0, 0.4, 0.8])

        def radiance(xy, period=period, phases=phases):
            angle = 2.0 * np.pi * xy / period
            base = np.sin(angle[..., 0]) * np.sin(angle[..., 1])
            return 0.5 + 0.5 * np.sin(base[..., None] * np.pi + phases)

        rows, cols = geometry.packed_shape
        for channel in range(geometry.channels):
            for row in range(rows):
                for col in range(cols):
                    sensor_y, sensor_x = geometry.sample_sensor_yx(channel, row, col)
                    point = np.array([sensor_x + 1.5, sensor_y - 0.5], dtype=np.float64)
                    color = radiance(point)
                    if geometry.is_cfa:
                        name = CFA_PATTERN_COLORS[geometry.cfa_pattern][
                            (sensor_y % 2) * 2 + (sensor_x % 2)
                        ]
                        expected = calibration.black_level_dn + 1400.0 * color[COLOR_INDEX[name]]
                    else:
                        expected = calibration.black_level_dn + 1400.0 * color[channel]
                    assert float(rendered.values[channel, row, col]) == pytest.approx(
                        expected, rel=2e-5, abs=2e-3
                    )


def test_exposure_gain_scales_the_value_between_frames():
    geometry = small_geometry()
    layers = (Layer("background", SinusoidField(period_xy=(9.0, 7.0)), SimilarityPose()),)
    dim = SceneFrame(frame_id=0, time=0.0, layers=layers, exposure_gain=1000.0)
    bright = SceneFrame(frame_id=1, time=1.0, layers=layers, exposure_gain=2000.0)
    calibration = _calibration()
    common = {
        "dataset_id": "d",
        "episode_id": "e",
        "camera_id": "c",
        "add_noise": False,
    }
    dim_values = render_raw_frame(dim, geometry, calibration, **common).values
    bright_values = render_raw_frame(bright, geometry, calibration, **common).values
    black = calibration.black_level_dn
    assert torch.allclose(
        bright_values.double() - black, 2.0 * (dim_values.double() - black), atol=2e-3
    )
    assert not torch.allclose(bright_values, dim_values)


def test_injected_noise_follows_the_declared_calibration():
    """A constant-radiance frame isolates the noise process from the geometry.

    The shared block error is constant inside one frame, so the frame mean of
    ``noisy - clean`` must equal the independently drawn block error, while the
    frame variance must equal ``shot * signal + read_var``.  An implementation
    that drew an independent offset per sample, or used the wrong variance,
    fails both halves.
    """
    from owac.raw.n01_noise import draw_block_error

    # The shipped 64x64 sensor grid gives 4,096 samples per frame, so the frame mean
    # of the read noise is tight enough to identify the shared block error.
    geometry = FrameGeometry(height=64, width=64)
    field = BlobField(centers_xy=((0.0, 0.0),), sigma=1.0e6, colors_rgb=((0.5, 0.4, 0.3),))
    calibration = _calibration()
    seed = 424242
    for gain in (1000.0, 2000.0):
        layer = Layer("background", field, SimilarityPose())
        scene = SceneFrame(frame_id=5, time=5.0, layers=(layer,), exposure_gain=gain)
        clean = render_raw_frame(
            scene,
            geometry,
            calibration,
            dataset_id="d",
            episode_id="e",
            camera_id="c",
            add_noise=False,
        ).values.double()
        frame = render_raw_frame(
            scene,
            geometry,
            calibration,
            dataset_id="d",
            episode_id="e",
            camera_id="c",
            add_noise=True,
            noise_seed=seed,
        )
        observed = frame.values.double() - clean
        block_id = calibration.black_level.id_template.format(
            dataset_id="d", episode_id="e", camera_id="c", frame_id=5
        )
        expected_block = draw_block_error(
            block_id, calibration, torch.Generator().manual_seed(seed)
        )
        law = calibration.shot_read.variance(clean, black_level=calibration.black_level_dn)
        # Five sigma of the read-noise frame mean, so the check is about the block
        # error being shared and correctly drawn, not about a lucky draw.
        tolerance = 5.0 * math.sqrt(float(law.mean()) / law.numel())
        assert float(observed.mean()) == pytest.approx(expected_block, abs=tolerance)
        assert tolerance < abs(expected_block)
        centred = observed - observed.mean()
        assert float(centred.pow(2).mean()) == pytest.approx(float(law.mean()), rel=0.05)
        assert frame.values.dtype == torch.float32
        assert frame.validity is not None and bool(frame.validity.all())


def test_motion_blur_integrates_the_exposure_quadrature():
    geometry = small_geometry()
    field = SinusoidField(period_xy=(9.0, 7.0))
    blur = 3.0
    samples = 5
    sharp = SceneFrame(
        frame_id=0,
        time=0.0,
        layers=(Layer("background", field, SimilarityPose(translation_xy=(0.5, 0.0))),),
        exposure_gain=1400.0,
    )
    blurred = SceneFrame(
        frame_id=1,
        time=1.0,
        layers=(Layer("background", field, SimilarityPose(translation_xy=(0.5, 0.0))),),
        exposure_gain=1400.0,
        blur_scene_displacement=(blur, 0.0),
        blur_samples=samples,
    )
    calibration = _calibration()
    common = {
        "dataset_id": "d",
        "episode_id": "e",
        "camera_id": "c",
        "add_noise": False,
    }
    sharp_values = render_raw_frame(sharp, geometry, calibration, **common).values
    blurred_values = render_raw_frame(blurred, geometry, calibration, **common).values
    assert not torch.allclose(sharp_values, blurred_values)
    offsets = np.linspace(-0.5, 0.5, samples)
    rows, cols = geometry.packed_shape
    for channel in range(geometry.channels):
        sensor_y, sensor_x = geometry.sample_sensor_yx(channel, rows // 2, cols // 2)
        color = (
            CFA_PATTERN_COLORS[geometry.cfa_pattern][(sensor_y % 2) * 2 + (sensor_x % 2)]
            if geometry.is_cfa
            else None
        )
        total = 0.0
        for offset in offsets:
            point = np.array([sensor_x + 0.5 - blur * offset, sensor_y], dtype=np.float64)
            angle = 2.0 * np.pi * point / np.array([9.0, 7.0])
            base = np.sin(angle[0]) * np.sin(angle[1])
            value = 0.5 + 0.5 * np.sin(base * np.pi + np.array([0.0, 0.4, 0.8]))
            total += value[COLOR_INDEX[color]] if color else value[channel]
        expected = calibration.black_level_dn + 1400.0 * total / samples
        assert float(blurred_values[channel, rows // 2, cols // 2]) == pytest.approx(
            expected, rel=2e-5, abs=2e-3
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_shipped_config_loads_and_covers_every_declared_case():
    config = load_p2_motion_noise_config(CONFIG_PATH)
    assert config.schema == MOTION_SCHEMA
    assert tuple(sorted(config.cases)) == tuple(sorted(CASE_KINDS))
    assert tuple(config.regions) == REGIONS
    assert config.geometry.is_cfa and config.pseudo_raw_geometry.input_mode == "rawvla_rgb3"
    assert config.geometry_for("synthetic_cfa") is config.geometry
    assert config.geometry_for("rawvla_rgb3") is config.pseudo_raw_geometry
    assert config.policy.fitting_splits == ("train", "calibration")
    assert set(config.materials) == {
        CFA_MATERIAL_CLASS,
        PSEUDO_RAW_MATERIAL_CLASS,
        "inverted_rgb_capture",
    }
    assert (
        config.to_dict()
        == P2MotionNoiseConfig.from_dict(json.loads(json.dumps(config.to_dict()))).to_dict()
    )


def test_scene_recipe_round_trips_every_declared_field():
    """A non-default value in every field must survive the JSON round trip.

    This is the guard that caught a field the loader did not read back: a
    silently dropped key would make the config JSON decorative.
    """
    recipe = SceneRecipe(
        background="steps",
        object_field="sinusoid",
        occluder=True,
        occluder_radius=3.5,
        occluder_scale_reference=1.25,
        background_step_px=(0.75, -0.25),
        object_step_px=(-1.5, 2.5),
        rotation_deg=1.25,
        warp_amplitude_px=2.75,
        warp_sigma_px=5.5,
        warp_ramp_frames=3,
        gain_reference=1234.0,
        gain_source=987.0,
        exposure_known=False,
        blur_length_px=1.75,
        blur_samples=6,
        texture_period_px=11.5,
        jitter_px=0.125,
        sequence_convention="round-trip probe",
    )
    restored = SceneRecipe.from_dict(json.loads(json.dumps(recipe.to_dict())))
    assert restored == recipe
    assert restored != SceneRecipe()


def test_config_rejects_drift_and_unknown_fields(tmp_path):
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dropped = json.loads(json.dumps(payload))
    del dropped["cases"]["motion_blur"]
    with pytest.raises(ValueError, match="cases must cover exactly"):
        P2MotionNoiseConfig.from_dict(dropped)
    unknown = json.loads(json.dumps(payload))
    unknown["cases"]["rotation"]["blur_radius"] = 1.0
    with pytest.raises(ValueError, match="unknown fields"):
        P2MotionNoiseConfig.from_dict(unknown)
    top_level = json.loads(json.dumps(payload))
    top_level["sensor"] = "unknown"
    with pytest.raises(ValueError, match="unknown fields"):
        P2MotionNoiseConfig.from_dict(top_level)
    wrong_schema = json.loads(json.dumps(payload))
    wrong_schema["schema"] = "owac.n01_full.p2_motion_noise.v0"
    with pytest.raises(ValueError, match="unknown schema"):
        P2MotionNoiseConfig.from_dict(wrong_schema)
    bad_geometry = json.loads(json.dumps(payload))
    bad_geometry["geometry"]["height"] = 65
    with pytest.raises(ValueError, match="even"):
        P2MotionNoiseConfig.from_dict(bad_geometry)
    bad_pseudo = json.loads(json.dumps(payload))
    bad_pseudo["pseudo_raw_geometry"]["channel_phase"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="no CFA phase"):
        P2MotionNoiseConfig.from_dict(bad_pseudo)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_p2_motion_noise_config(path).config_version == payload["config_version"]


def test_repeated_texture_case_declares_its_ambiguity_scale(generator):
    pair = generator.sample_pair(split="train", case="repeated_texture", sequence="0")
    assert pair.supervision.ambiguity_period_px == pytest.approx(
        generator.config.recipe("repeated_texture").texture_period_px
    )
    other = generator.sample_pair(split="train", case="rotation", sequence="0")
    assert other.supervision.ambiguity_period_px is None


def test_packed_pull_targets_refuses_channel_mismatch():
    geometry = small_geometry()
    other = FrameGeometry(height=16, width=16, origin_yx=(1, 0), channel_phase=(2, 3, 0, 1))
    source = SceneFrame(
        frame_id=0,
        time=0.0,
        layers=(Layer("background", SinusoidField(), SimilarityPose()),),
    )
    reference = SceneFrame(
        frame_id=1,
        time=1.0,
        layers=(Layer("background", SinusoidField(), SimilarityPose()),),
    )
    solve = solve_pull(pair_layers(reference, source), geometry, geometry)
    assert packed_pull_targets(solve, geometry, geometry).shape == (4, 8, 8, 2)
    with pytest.raises(ValueError, match="channel_phase"):
        packed_pull_targets(solve, geometry, other)

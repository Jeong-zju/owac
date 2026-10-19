"""P1 observation layer tests: multi-scale RAW encoding, coordinates and FCMAE masks.

These tests assert real numbers and real causality, not imports:

* ``raw3``/``cfa4`` level shapes, pointwise validity and pointwise sensor coordinates,
  including the packed-256 vs sensor-512 distinction and exact coordinate round trips;
* rectangular (non-square) frames: stride-aligned padding is recorded, coordinates survive
  the mapping and padded cells are invalid and exactly zero;
* the encoder path uses the P0-parity ConvNeXt V2 backbone: the exported fixture reference
  is reproduced and the projected level equals an independent recomputation;
* FCMAE masks are reproducible from identity, depend on the current frame's support, and
  are unchanged by pixel content, other cameras, later frames or truth fields;
* the FCMAE loss counts only valid, non-saturated target samples and its gradients react to
  injected missing/bad/saturated samples;
* structurally, no reconstruction head and no RGB path exist on the observation encoder.

All tests run on CPU so the shared GPU (a frozen baseline job) is not touched.

One comparison is deliberately **not** a strict bitwise claim: feature ``values`` across
different batch sizes. See :func:`_assert_row_matches_single_row` for the dispatcher-accepted
contract (bitwise ``valid``/coordinates, ``1e-5`` magnitude bound for values) and the measured
divergence that motivates it. Do not read the batch-axis tests as proving strict bitwise
equality of feature values.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.components.convnextv2 import RawConvNeXtV2
from owac.frameworks.n02_full.components.fcmae import gen_random_mask, upsample_patch_mask
from owac.frameworks.n02_full.observation import (
    CFA4,
    FCMAE_MASK_RATIO,
    FCMAE_PATCH_SIZE,
    PROJECTION_DIM,
    RAW3,
    FCMAEOutput,
    FeaturePyramid,
    RawFCMAEPretrainer,
    RawObservationEncoder,
    frame_transform,
    mask_seed,
    packed_to_sensor_coordinates,
    sensor_to_packed_coordinates,
)
from owac.frameworks.n02_full.state import FeatureLevel, ObservationBundle

FIXTURE_DIR = Path("outputs/n02-full/p0/parity/fixtures")
DEVICE = torch.device("cpu")

# Small but real four-stage backbone: the tests exercise the code path, not the Base budget.
SMALL_DEPTHS = (1, 1, 1, 1)
SMALL_DIMS = (8, 16, 32, 64)


def _fixture(name: str):
    path = FIXTURE_DIR / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"P0 reference fixture {path} not present")
    return torch.load(path, map_location="cpu", weights_only=False)


def _encoder(mode: str, **kwargs) -> RawObservationEncoder:
    encoder = RawObservationEncoder(input_mode=mode, depths=SMALL_DEPTHS, dims=SMALL_DIMS, **kwargs)
    return encoder.to(DEVICE).eval()


def _pretrainer(**kwargs) -> RawFCMAEPretrainer:
    pretrainer = RawFCMAEPretrainer(
        input_mode=RAW3, depths=SMALL_DEPTHS, dims=SMALL_DIMS, img_size=(96, 96), **kwargs
    )
    return pretrainer.to(DEVICE).eval()


def _frame(
    values,
    *,
    camera_id: str = "cam0",
    episode_ids: tuple[str, ...] = ("ep0",),
    frame_ids: tuple[str, ...] = ("f0",),
    valid_mask=None,
    bad_mask=None,
    saturated_mask=None,
    metadata=None,
) -> MeasurementFrame:
    return MeasurementFrame(
        values=values,
        valid_mask=valid_mask,
        bad_mask=bad_mask,
        saturated_mask=saturated_mask,
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        camera_id=camera_id,
        metadata=metadata if metadata is not None else {},
    )


def _bundle(frames: dict, *, episode_ids=("ep0",)) -> ObservationBundle:
    return ObservationBundle(frames=frames, context=EmbodimentContext(), episode_ids=episode_ids)


# --------------------------------------------------------------------------------------
# steps 1: shapes, validity and sensor coordinates in both modes
# --------------------------------------------------------------------------------------


def test_raw3_levels_shapes_validity_and_sensor_coordinates_pointwise():
    encoder = _encoder(RAW3)
    torch.manual_seed(7)
    frame = _frame(torch.rand(2, 3, 64, 96), episode_ids=("ep0", "ep1"), frame_ids=("f0", "f1"))
    pyramid = encoder.encode(_bundle({"cam0": frame}))

    assert set(pyramid.levels) == {"stride8", "stride16"}
    assert pyramid.input_mode == RAW3
    assert pyramid.camera_ids == ("cam0",)
    assert pyramid.sensor_shape == {"cam0": (64, 96)}

    level8 = pyramid.levels["stride8"]
    level16 = pyramid.levels["stride16"]
    assert level8.values.shape == (2, PROJECTION_DIM, 8, 12)
    assert level16.values.shape == (2, PROJECTION_DIM, 4, 6)
    assert level8.valid.shape == (2, 8, 12) and level8.valid.dtype == torch.bool
    assert level8.valid.all() and level16.valid.all()
    assert level8.level_id == "stride8" and level16.level_id == "stride16"
    assert level8.support_ids == ("ep0|cam0|f0", "ep1|cam0|f1")

    coordinates = level8.sensor_coordinates
    assert coordinates.shape == (2, 8, 12, 2)
    assert coordinates.dtype == torch.float32
    rows = (torch.arange(8, dtype=torch.float32) + 0.5) * 8.0
    cols = (torch.arange(12, dtype=torch.float32) + 0.5) * 8.0
    assert torch.equal(coordinates[0, :, :, 0], rows[:, None].expand(8, 12))
    assert torch.equal(coordinates[0, :, :, 1], cols[None, :].expand(8, 12))
    assert torch.equal(coordinates[1], coordinates[0])

    rows16 = (torch.arange(4, dtype=torch.float32) + 0.5) * 16.0
    cols16 = (torch.arange(6, dtype=torch.float32) + 0.5) * 16.0
    assert torch.equal(level16.sensor_coordinates[0, :, :, 0], rows16[:, None].expand(4, 6))
    assert torch.equal(level16.sensor_coordinates[0, :, :, 1], cols16[None, :].expand(4, 6))

    transform = pyramid.transforms["cam0"]
    assert transform.pad_after == (0, 0) and transform.pad_before == (0, 0)
    assert transform.frame_shape == (64, 96) and transform.padded_shape == (64, 96)
    assert transform.packed_scale == 1


def test_missing_pixel_invalidates_exactly_the_cells_that_cover_it():
    encoder = _encoder(RAW3)
    torch.manual_seed(5)
    values = torch.rand(1, 3, 64, 96)
    full = encoder.encode(_bundle({"cam0": _frame(values)}))

    # (a) a hole inside a window keeps the cell supported, but the hole is zero-filled:
    #     the feature values change and never fabricate a measurement
    hole = torch.ones(1, 64, 96, dtype=torch.bool)
    hole[0, 10, 20] = False
    with_hole = encoder.encode(_bundle({"cam0": _frame(values, valid_mask=hole)}))
    assert torch.equal(with_hole.levels["stride8"].valid, full.levels["stride8"].valid)
    assert not torch.equal(with_hole.levels["stride8"].values, full.levels["stride8"].values)

    # (b) a fully unsupported 16x16 block removes exactly the cells whose whole window is
    #     unsupported (the P0 dilate_support rule: support exists iff the window keeps a
    #     measurement), and those cells are exactly zero on every channel
    blocked = torch.ones(1, 64, 96, dtype=torch.bool)
    blocked[0, 16:32, 16:32] = False
    dropped = encoder.encode(_bundle({"cam0": _frame(values, valid_mask=blocked)}))
    invalid8 = (~dropped.levels["stride8"].valid[0]).nonzero().tolist()
    assert invalid8 == [[2, 2], [2, 3], [3, 2], [3, 3]]
    invalid16 = (~dropped.levels["stride16"].valid[0]).nonzero().tolist()
    assert invalid16 == [[1, 1]]
    for row, column in invalid8:
        assert (
            float(dropped.levels["stride8"].values[0][:, row, column].detach().abs().max()) == 0.0
        )
    assert float(dropped.levels["stride16"].values[0][:, 1, 1].detach().abs().max()) == 0.0
    # neighbouring cells keep their (non-zero) values
    assert float(dropped.levels["stride8"].values[0][:, 1, 2].detach().abs().max()) > 0.0


def test_cfa4_packed_256_grid_reports_sensor_512_coordinates():
    encoder = _encoder(CFA4)
    torch.manual_seed(11)
    mosaic = torch.rand(1, 512, 512)  # one Bayer mosaic plane; identity via the pattern
    frame = _frame(mosaic, metadata={"camera": {"cfa_pattern": "GRBG"}})
    pyramid = encoder.encode(_bundle({"cam0": frame}))

    level8 = pyramid.levels["stride8"]
    # a 512x512 sensor crop becomes a 256x256 packed grid, then stride 8 -> 32x32 cells
    assert level8.values.shape == (1, PROJECTION_DIM, 32, 32)
    assert pyramid.levels["stride16"].values.shape == (1, PROJECTION_DIM, 16, 16)
    # the sensor resolution is 512x512 and is never reported as the packed 256
    assert pyramid.sensor_shape == {"cam0": (512, 512)}
    assert pyramid.transforms["cam0"].frame_shape == (256, 256)
    assert pyramid.transforms["cam0"].packed_scale == 2

    coordinates = level8.sensor_coordinates[0]
    assert float(coordinates[..., 0].min()) == 8.0
    assert float(coordinates[..., 0].max()) == 504.0
    assert float(coordinates[..., 1].max()) == 504.0
    step = float(coordinates[1, 0, 0] - coordinates[0, 0, 0])
    assert step == 16.0  # sensor stride 2 x 8 packed cells
    expected = (torch.arange(32, dtype=torch.float32) + 0.5) * 16.0
    assert torch.equal(coordinates[:, 0, 0], expected)
    assert torch.equal(coordinates[0, :, 1], expected)
    # every coordinate maps back to the level cell centre in packed-cell units
    cells = sensor_to_packed_coordinates(coordinates, mode=CFA4)
    expected_cells = (torch.arange(32, dtype=torch.float64) + 0.5) * 8.0 - 0.5
    assert torch.allclose(cells[:, 0, 0], expected_cells)
    assert torch.allclose(cells[0, :, 1], expected_cells)


def test_packed_sensor_coordinate_roundtrip_is_exact_even_with_padding_and_crop():
    indices = torch.arange(0, 64, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(indices, indices, indexing="ij"), dim=-1)

    for mode, scale in ((RAW3, 1), (CFA4, 2)):
        sensor = packed_to_sensor_coordinates(grid, mode=mode)
        assert torch.equal(sensor[..., 0], scale * (grid[..., 0] + 0.5))
        assert torch.equal(sensor[..., 1], scale * (grid[..., 1] + 0.5))
        assert torch.equal(sensor_to_packed_coordinates(sensor, mode=mode), grid)

    sensor = packed_to_sensor_coordinates(
        grid, mode=CFA4, pad_before=(3, 5), crop_offset=(100, 200)
    )
    assert float(sensor[0, 0, 0]) == 100.0 + 2.0 * (3 + 0.5)
    assert float(sensor[0, 0, 1]) == 200.0 + 2.0 * (5 + 0.5)
    back = sensor_to_packed_coordinates(
        sensor, mode=CFA4, pad_before=(3, 5), crop_offset=(100, 200)
    )
    assert torch.equal(back, grid)
    # a non-integer sensor coordinate is still inverted consistently (continuity)
    probe = torch.tensor(
        [[[100.0 + 2.0 * (3 + 7.3 + 0.5), 200.0 + 2.0 * (5 + 1.25 + 0.5)]]], dtype=torch.float64
    )
    inverted = sensor_to_packed_coordinates(
        probe, mode=CFA4, pad_before=(3, 5), crop_offset=(100, 200)
    )
    assert abs(float(inverted[0, 0, 0]) - 7.3) < 1e-12
    assert abs(float(inverted[0, 0, 1]) - 1.25) < 1e-12


# --------------------------------------------------------------------------------------
# steps 2: rectangular frames, padding record and coordinates
# --------------------------------------------------------------------------------------


def test_rectangular_raw3_padding_is_recorded_and_coordinates_are_kept():
    encoder = _encoder(RAW3)
    torch.manual_seed(13)
    frame = _frame(torch.rand(1, 3, 240, 320), metadata={"camera": {"crop_offset": [4, 6]}})
    pyramid = encoder.encode(_bundle({"cam0": frame}))

    transform = pyramid.transforms["cam0"]
    assert transform.frame_shape == (240, 320)
    assert transform.padded_shape == (256, 320)
    assert transform.pad_before == (0, 0)
    assert transform.pad_after == (16, 0)
    assert transform.crop_offset == (4, 6)
    assert transform.sensor_shape == (244, 326)
    assert pyramid.sensor_shape == {"cam0": (244, 326)}

    level8 = pyramid.levels["stride8"]
    assert level8.values.shape == (1, PROJECTION_DIM, 32, 40)
    # rows 0..29 cover the real 240 rows, rows 30..31 only the padding
    assert bool(level8.valid[0, :30, :].all())
    assert not bool(level8.valid[0, 30:, :].any())
    assert float(level8.values[0][:, 30:, :].detach().abs().max()) == 0.0

    coordinates = level8.sensor_coordinates[0]
    assert coordinates[0, 0].tolist() == [4.0 + 4.0, 6.0 + 4.0]
    assert float(coordinates[31, 0, 0]) == 4.0 + 8.0 * 31.5
    assert float(coordinates[0, 39, 1]) == 6.0 + 8.0 * 39.5
    # no coordinate is lost: the grid is complete and strictly increasing
    assert coordinates.shape[:2] == (32, 40)
    assert bool((coordinates[1:, :, 0] > coordinates[:-1, :, 0]).all())
    assert bool((coordinates[:, 1:, 1] > coordinates[:, :-1, 1]).all())


def test_rectangular_cfa4_padding_is_recorded_in_packed_cells():
    encoder = _encoder(CFA4)
    torch.manual_seed(17)
    mosaic = torch.rand(1, 240, 320)
    frame = _frame(mosaic, metadata={"camera": {"cfa_pattern": "RGGB"}})
    pyramid = encoder.encode(_bundle({"cam0": frame}))

    transform = pyramid.transforms["cam0"]
    assert transform.frame_shape == (120, 160)
    assert transform.padded_shape == (128, 160)
    assert transform.pad_after == (8, 0)
    assert transform.sensor_shape == (240, 320)

    level8 = pyramid.levels["stride8"]
    assert level8.values.shape == (1, PROJECTION_DIM, 16, 20)
    assert bool(level8.valid[0, :15, :].all())
    assert not bool(level8.valid[0, 15:, :].any())
    assert float(level8.values[0][:, 15:, :].detach().abs().max()) == 0.0

    coordinates = level8.sensor_coordinates[0]
    assert float(coordinates[0, 0, 0]) == 8.0
    assert float(coordinates[0, 0, 1]) == 8.0
    assert float(coordinates[14, 0, 0]) == 2.0 * 8.0 * 14.5  # last real packed row
    assert float(coordinates[15, 0, 0]) == 2.0 * 8.0 * 15.5  # padded row still mapped


def test_cfa4_batch_axis_is_batch_first_and_each_row_matches_a_single_row_run():
    """Freeze section 3.9: a batched CFA4 mosaic frame is ``(B, 1, H, W)``, never ``(C, H, W)``.

    The pre-fix code read ``(2, H, W)`` as two channels and raised
    ``cfa4 mode accepts 1 mosaic channel or 4 packed CFA planes, got 2``, which forced the
    training entry to run one episode at a time.
    """
    encoder = _encoder(CFA4)
    torch.manual_seed(79)
    device_metadata = {"camera": {"cfa_pattern": "RGGB"}}
    values = torch.rand(2, 240, 320)  # (B, H, W): two mosaic rows
    ids = (("ep0", "ep1"), ("f0", "f1"))
    batched = encoder.encode(
        _bundle(
            {
                "cam0": _frame(
                    values,
                    episode_ids=ids[0],
                    frame_ids=ids[1],
                    metadata=device_metadata,
                )
            },
            episode_ids=ids[0],
        )
    )

    assert batched.sensor_shape == {"cam0": (240, 320)}
    assert batched.transforms["cam0"].frame_shape == (120, 160)
    assert batched.transforms["cam0"].pad_after == (8, 0)
    for level_id, shape in (
        ("stride8", (2, PROJECTION_DIM, 16, 20)),
        ("stride16", (2, PROJECTION_DIM, 8, 10)),
    ):
        level = batched.levels[level_id]
        assert level.values.shape == shape
        assert level.valid.shape == shape[0:1] + shape[2:]
        assert level.sensor_coordinates.shape == shape[0:1] + shape[2:] + (2,)

    # every row of the batch equals the same row encoded alone
    for row in range(2):
        single = encoder.encode(
            _bundle(
                {
                    "cam0": _frame(
                        values[row : row + 1],
                        episode_ids=(f"ep{row}",),
                        frame_ids=(f"f{row}",),
                        metadata=device_metadata,
                    )
                },
                episode_ids=(f"ep{row}",),
            )
        )
        for level_id in ("stride8", "stride16"):
            _assert_row_matches_single_row(
                batched.levels[level_id], single.levels[level_id], row, label=f"{level_id}/row{row}"
            )
        assert single.levels["stride8"].support_ids == (f"ep{row}|cam0|f{row}",)
    assert batched.levels["stride8"].support_ids == ("ep0|cam0|f0", "ep1|cam0|f1")

    # batch rows are independent: replacing row 1's pixels leaves row 0's features bitwise
    # identical (the dense backbone couples space per sample, never across batch rows)
    replaced = values.clone()
    replaced[1] = torch.rand(240, 320)
    independent = encoder.encode(
        _bundle(
            {
                "cam0": _frame(
                    replaced,
                    episode_ids=ids[0],
                    frame_ids=ids[1],
                    metadata=device_metadata,
                )
            },
            episode_ids=ids[0],
        )
    )
    for level_id in ("stride8", "stride16"):
        assert torch.equal(
            batched.levels[level_id].values[0:1], independent.levels[level_id].values[0:1]
        ), level_id
        assert not torch.equal(
            batched.levels[level_id].values[1:2], independent.levels[level_id].values[1:2]
        ), level_id

    # a batched mosaic still needs its declared Bayer pattern, and a batched packed frame
    # keeps the explicit channel axis
    with pytest.raises(ValueError, match="Bayer pattern"):
        encoder.encode(_bundle({"cam0": _frame(values, episode_ids=ids[0], frame_ids=ids[1])}))
    packed = encoder.encode(
        _bundle(
            {
                "cam0": _frame(
                    torch.rand(2, 4, 32, 48),
                    episode_ids=ids[0],
                    frame_ids=ids[1],
                    metadata=device_metadata,
                )
            },
            episode_ids=ids[0],
        )
    )
    assert packed.levels["stride8"].values.shape == (2, PROJECTION_DIM, 4, 8)
    assert packed.sensor_shape == {"cam0": (64, 96)}

    # B == 4 collides with the packed channel count: the declared batch rows win, so four
    # mosaic rows stay four rows instead of becoming one packed 4-plane frame
    four = torch.rand(4, 128, 128)
    four_rows = encoder.encode(
        _bundle(
            {
                "cam0": _frame(
                    four,
                    episode_ids=("e0", "e1", "e2", "e3"),
                    frame_ids=("f0", "f1", "f2", "f3"),
                    metadata=device_metadata,
                )
            },
            episode_ids=("e0", "e1", "e2", "e3"),
        )
    )
    assert four_rows.levels["stride8"].values.shape == (4, PROJECTION_DIM, 8, 8)
    # without declared rows the same 3-D shape is the single-sample packed spelling
    packed_single = encoder.encode(
        _bundle({"cam0": _frame(four, metadata=device_metadata)}, episode_ids=("e0",))
    )
    assert packed_single.levels["stride8"].values.shape == (1, PROJECTION_DIM, 16, 16)


def test_cfa4_batched_multicamera_heterogeneous_resolutions_keep_the_batch_axis():
    encoder = _encoder(CFA4)
    torch.manual_seed(83)
    metadata = {"camera": {"cfa_pattern": "GRBG"}}
    values_a = torch.rand(2, 240, 320)
    values_b = torch.rand(2, 128, 192)
    episodes = ("ep0", "ep1")
    frames = ("f0", "f1")
    pyramid = encoder.encode(
        _bundle(
            {
                "a": _frame(
                    values_a,
                    camera_id="a",
                    episode_ids=episodes,
                    frame_ids=frames,
                    metadata=metadata,
                ),
                "b": _frame(
                    values_b,
                    camera_id="b",
                    episode_ids=episodes,
                    frame_ids=frames,
                    metadata=metadata,
                ),
            },
            episode_ids=episodes,
        )
    )

    assert set(pyramid.levels) == {"a::stride8", "a::stride16", "b::stride8", "b::stride16"}
    assert pyramid.sensor_shape == {"a": (240, 320), "b": (128, 192)}
    grouped = pyramid.per_camera()
    assert grouped["a"]["stride8"].values.shape == (2, PROJECTION_DIM, 16, 20)
    assert grouped["b"]["stride8"].values.shape == (2, PROJECTION_DIM, 8, 12)
    assert grouped["b"]["stride16"].values.shape == (2, PROJECTION_DIM, 4, 6)

    single_encoder = encoder
    for camera_id, values in (("a", values_a), ("b", values_b)):
        for row in range(2):
            single = single_encoder.encode(
                _bundle(
                    {
                        camera_id: _frame(
                            values[row : row + 1],
                            camera_id=camera_id,
                            episode_ids=(f"ep{row}",),
                            frame_ids=(f"f{row}",),
                            metadata=metadata,
                        )
                    },
                    episode_ids=(f"ep{row}",),
                )
            )
            for level_id in ("stride8", "stride16"):
                _assert_row_matches_single_row(
                    grouped[camera_id][level_id],
                    single.levels[level_id],
                    row,
                    label=f"{camera_id}/{level_id}/row{row}",
                )


def test_cfa4_micro_batch_of_eight_mosaics_reaches_encoder_and_pretrainer():
    """The micro-batch the batch-axis defect blocked: eight CFA4 mosaic rows in one batch."""
    encoder = _encoder(CFA4)
    pretrainer = RawFCMAEPretrainer(
        input_mode=CFA4, depths=SMALL_DEPTHS, dims=SMALL_DIMS, img_size=(256, 256)
    ).eval()
    torch.manual_seed(89)
    values = torch.rand(8, 240, 320)
    episodes = tuple(f"ep{index}" for index in range(8))
    frames = tuple(f"f{index}" for index in range(8))
    frame = _frame(
        values,
        episode_ids=episodes,
        frame_ids=frames,
        metadata={"camera": {"cfa_pattern": "GRBG"}},
    )

    pyramid = encoder.encode(_bundle({"cam0": frame}, episode_ids=episodes))
    assert pyramid.levels["stride8"].values.shape == (8, PROJECTION_DIM, 16, 20)
    assert pyramid.levels["stride16"].values.shape == (8, PROJECTION_DIM, 8, 10)
    assert pyramid.transforms["cam0"].pad_after == (8, 0)
    assert bool(pyramid.levels["stride8"].valid[0, :15, :].all())
    assert not bool(pyramid.levels["stride8"].valid[0, 15:, :].any())

    out = pretrainer(frames={"cam0": frame}, camera_ids=("cam0",))
    # packed 120x160 -> padded 128x160 -> a 4x5 patch grid; the padded patch row has no
    # support, so its samples never enter the loss and the count is recomputed here
    assert out.mask.shape == (1, 8, 20)
    assert int((out.mask[0, 0] == 1).sum()) == 20 - int(20 * (1 - 0.6))
    support = torch.ones(8, 128, 160, dtype=torch.bool)
    support[:, 120:, :] = False
    targets = upsample_patch_mask(out.mask[0], 32, (4, 5)).unsqueeze(1)
    expected = int((targets * support.unsqueeze(1)).sum().item()) * 4
    assert int(out.valid_target_count.item()) == expected
    assert int(out.saturated_count.item()) == 0
    assert torch.isfinite(out.loss)


# --------------------------------------------------------------------------------------
# steps 3: multi-camera identity and the fixture-backed numerical reference
# --------------------------------------------------------------------------------------


def test_multicamera_keeps_camera_identity_resolution_and_coordinates():
    encoder = _encoder(RAW3)
    torch.manual_seed(23)
    frame_a = _frame(
        torch.rand(1, 3, 64, 64), camera_id="a", episode_ids=("ep0",), frame_ids=("fa",)
    )
    frame_b = _frame(
        torch.rand(1, 3, 32, 96), camera_id="b", episode_ids=("ep0",), frame_ids=("fb",)
    )
    pyramid = encoder.encode(_bundle({"a": frame_a, "b": frame_b}))

    assert set(pyramid.levels) == {"a::stride8", "a::stride16", "b::stride8", "b::stride16"}
    grouped = pyramid.per_camera()
    assert grouped["a"]["stride8"].values.shape == (1, PROJECTION_DIM, 8, 8)
    assert grouped["b"]["stride8"].values.shape == (1, PROJECTION_DIM, 4, 12)
    assert grouped["b"]["stride16"].values.shape == (1, PROJECTION_DIM, 2, 6)
    assert pyramid.sensor_shape == {"a": (64, 64), "b": (32, 96)}

    tokens = pyramid.tokens()
    expected_tokens = 8 * 8 + 4 * 4 + 4 * 12 + 2 * 6
    assert tokens.values.shape == (1, expected_tokens, PROJECTION_DIM)
    assert tokens.sensor_coordinates.shape == (1, expected_tokens, 2)
    assert tokens.valid.all()
    camera_index = tokens.camera_index[0]
    level_index = tokens.level_index[0]
    # camera identity is carried per token, never folded into the batch axis
    assert tokens.values.shape[0] == 1
    assert camera_index[: 8 * 8 + 4 * 4].unique().tolist() == [0]
    assert camera_index[8 * 8 + 4 * 4 :].unique().tolist() == [1]
    assert level_index[: 8 * 8].unique().tolist() == [0]
    assert level_index[8 * 8 : 8 * 8 + 4 * 4].unique().tolist() == [1]
    assert tokens.camera_ids == ("a", "b")
    first = grouped["b"]["stride8"].sensor_coordinates[0, 0, 0]
    assert torch.equal(tokens.sensor_coordinates[0, 8 * 8 + 4 * 4], first)


def test_support_hash_is_deterministic_support_sensitive_and_order_independent():
    encoder = _encoder(RAW3)
    torch.manual_seed(29)
    values = torch.rand(1, 3, 64, 96)
    frame = _frame(values, camera_id="a", frame_ids=("fa",))
    other = _frame(torch.rand(1, 3, 32, 96), camera_id="b", frame_ids=("fb",))

    first = encoder.encode(_bundle({"a": frame})).support_hash
    second = encoder.encode(_bundle({"a": frame})).support_hash
    assert first == second

    # camera dict order must not matter
    swapped = encoder.encode(_bundle({"b": other, "a": frame})).support_hash
    ordered = encoder.encode(_bundle({"a": frame, "b": other})).support_hash
    assert swapped == ordered

    support = torch.ones(1, 64, 96, dtype=torch.bool)
    support[0, 0, 0] = False
    changed = encoder.encode(
        _bundle({"a": _frame(values, camera_id="a", frame_ids=("fa",), valid_mask=support)})
    ).support_hash
    assert changed != first

    # pixel content is not part of the support hash
    other_values = encoder.encode(
        _bundle({"a": _frame(torch.rand(1, 3, 64, 96), camera_id="a", frame_ids=("fa",))})
    ).support_hash
    assert other_values == first


def test_backbone_matches_p0_fixture_and_encoder_recompute_is_exact():
    fixture = _fixture("convnext_4ch_rect")
    model = RawConvNeXtV2(
        in_chans=fixture["in_chans"],
        depths=fixture["depths"],
        dims=fixture["dims"],
        num_classes=fixture["num_classes"],
    ).eval()
    model.load_state_dict(fixture["state_dict"], strict=True)
    with torch.no_grad():
        reference = model.forward_multiscale(fixture["input"])
    for stage, expected in enumerate(fixture["stage_outputs"]):
        stride = 4 * 2**stage
        got = reference[f"stage{stage}_stride{stride}"]
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)

    dims = tuple(int(item) for item in fixture["dims"])
    encoder = RawObservationEncoder(
        input_mode=CFA4,
        in_chans=fixture["in_chans"],
        depths=tuple(int(item) for item in fixture["depths"]),
        dims=dims,
    ).eval()
    state = {
        key: value for key, value in fixture["state_dict"].items() if not key.startswith("head.")
    }
    encoder.backbone.load_state_dict(state, strict=True)

    # the fixture frame is 32x48; extend it on the right so the stride alignment pads nothing
    padded_input = torch.cat([fixture["input"], fixture["input"][:, :, :, :16]], dim=3)
    assert padded_input.shape[-2:] == (32, 64)
    frame = _frame(padded_input, metadata={"camera": {"cfa_pattern": "GBRG"}})
    with torch.no_grad():
        levels = encoder.encode_frame(frame, camera_id="cam0")

    with torch.no_grad():
        recomputed = model.forward_multiscale(padded_input)
    for level_id, stage, channels in (("stride8", 1, dims[1]), ("stride16", 2, dims[2])):
        projection = encoder.proj[level_id]
        assert projection.weight.shape == (PROJECTION_DIM, channels, 1, 1)
        expected = F.conv2d(
            recomputed[f"stage{stage}_stride{4 * 2**stage}"], projection.weight, projection.bias
        )
        level = levels[level_id]
        assert level.values.shape == (1, PROJECTION_DIM, *expected.shape[-2:])
        assert bool(level.valid.all())
        assert torch.equal(level.values, expected)
    # the two returned levels really are the two different stages
    assert levels["stride8"].values.shape[-2:] == (4, 8)
    assert levels["stride16"].values.shape[-2:] == (2, 4)


def test_encoding_is_deterministic_and_weight_sensitive():
    torch.manual_seed(31)
    frame = _frame(torch.rand(1, 3, 64, 96))
    torch.manual_seed(0)
    first = _encoder(RAW3)
    torch.manual_seed(0)
    second = _encoder(RAW3)
    torch.manual_seed(1)
    third = _encoder(RAW3)
    with torch.no_grad():
        base = first.encode(_bundle({"cam0": frame})).levels["stride8"].values
        repeat = second.encode(_bundle({"cam0": frame})).levels["stride8"].values
        different = third.encode(_bundle({"cam0": frame})).levels["stride8"].values
    assert torch.equal(base, repeat)
    assert float((base - different).detach().abs().max()) > 1e-6


def test_default_base_configuration_runs_and_projects_to_512():
    encoder = RawObservationEncoder(input_mode=RAW3).eval()
    assert encoder.depths == (3, 3, 27, 3)
    assert encoder.dims == (128, 256, 512, 1024)
    assert encoder.backbone.downsample_layers[0][0].in_channels == 3
    assert encoder.proj["stride8"].weight.shape == (PROJECTION_DIM, 256, 1, 1)
    assert encoder.proj["stride16"].weight.shape == (PROJECTION_DIM, 512, 1, 1)
    assert RawObservationEncoder(input_mode=CFA4).in_chans == 4

    torch.manual_seed(67)
    frame = _frame(torch.rand(1, 3, 64, 64))
    with torch.no_grad():
        pyramid = encoder.encode(_bundle({"cam0": frame}))
    assert pyramid.levels["stride8"].values.shape == (1, PROJECTION_DIM, 8, 8)
    assert pyramid.levels["stride16"].values.shape == (1, PROJECTION_DIM, 4, 4)
    assert bool(pyramid.levels["stride8"].valid.all())
    assert bool(pyramid.levels["stride16"].valid.all())


# --------------------------------------------------------------------------------------
# steps 4: declared calibration, mode separation and structural boundaries
# --------------------------------------------------------------------------------------


def test_calibration_normalisation_reproduces_a_precalibrated_encoding():
    torch.manual_seed(37)
    raw = torch.rand(1, 3, 64, 96) * 1000.0 + 100.0
    encoder = _encoder(RAW3)
    with torch.no_grad():
        declared = encoder.encode(
            _bundle(
                {
                    "cam0": _frame(
                        raw, metadata={"calibration": {"black_level": 100.0, "white_level": 1100.0}}
                    )
                }
            )
        )
        explicit = encoder.encode(_bundle({"cam0": _frame((raw - 100.0) / 1000.0)}))
    assert torch.equal(declared.levels["stride8"].values, explicit.levels["stride8"].values)
    assert torch.equal(declared.levels["stride16"].values, explicit.levels["stride16"].values)


def test_input_modes_are_explicit_and_never_converted():
    raw3 = _encoder(RAW3)
    cfa4 = _encoder(CFA4)
    assert raw3.backbone.downsample_layers[0][0].in_channels == 3
    assert cfa4.backbone.downsample_layers[0][0].in_channels == 4

    with pytest.raises(ValueError, match="cfa4"):
        raw3.encode(_bundle({"cam0": _frame(torch.rand(1, 1, 64, 64))}))
    with pytest.raises(ValueError, match="raw3"):
        cfa4.encode(_bundle({"cam0": _frame(torch.rand(1, 3, 64, 64))}))
    with pytest.raises(ValueError, match="Bayer pattern"):
        cfa4.encode(_bundle({"cam0": _frame(torch.rand(1, 1, 64, 64))}))
    with pytest.raises(ValueError, match="input_mode"):
        RawObservationEncoder(input_mode="rgb3")
    with pytest.raises(ValueError, match="input channels"):
        RawObservationEncoder(input_mode=CFA4, in_chans=3)


def test_encoder_has_no_reconstruction_or_rgb_path():
    encoder = _encoder(RAW3)
    names = (
        [name for name, _ in encoder.named_modules()]
        + [name for name, _ in encoder.named_parameters()]
        + [name for name, _ in encoder.named_buffers()]
    )
    assert names
    forbidden = ("decoder", "mask_token", "reconstruction", "rgb", "demosaic")
    assert not any(any(token in name for token in forbidden) for name in names)
    # the encoder has no single-tensor forward: only encode/encode_frame build observations
    with pytest.raises(NotImplementedError):
        encoder(torch.zeros(1, 3, 64, 64))

    assert set(FeaturePyramid.__dataclass_fields__) == {
        "levels",
        "camera_ids",
        "input_mode",
        "sensor_shape",
        "support_hash",
        "transforms",
    }
    assert set(FCMAEOutput.__dataclass_fields__) == {
        "loss",
        "reconstruction",
        "mask",
        "valid_target_count",
        "saturated_count",
    }
    # a pretrainer's decoder is a separate module tree, never reachable from the encoder
    pretrainer = _pretrainer()
    assert not hasattr(encoder, "fcmae")
    assert pretrainer is not encoder
    assert not any(
        str(parameter.data_ptr()) in {str(item.data_ptr()) for item in encoder.parameters()}
        for parameter in pretrainer.parameters()
    )


# --------------------------------------------------------------------------------------
# steps 5: FCMAE mask identity, reproducibility and causality
# --------------------------------------------------------------------------------------


def test_fcmae_frozen_hyperparameters_and_mask_reproducibility():
    pretrainer = _pretrainer()
    assert pretrainer.fcmae.mask_ratio == FCMAE_MASK_RATIO == 0.6
    assert pretrainer.patch_size == FCMAE_PATCH_SIZE == 32
    assert pretrainer.fcmae.decoder_embed_dim == PROJECTION_DIM == 512
    assert pretrainer.fcmae.decoder_depth == 1
    assert pretrainer.fcmae.norm_pix_loss is False
    assert pretrainer.in_chans == 3

    torch.manual_seed(41)
    values = torch.rand(1, 3, 96, 96)  # 3x3 patch grid, L = 9
    frame = _frame(values)
    first = pretrainer.generate_mask(frame, camera_id="cam0")
    second = pretrainer.generate_mask(frame, camera_id="cam0")
    assert torch.equal(first, second)
    assert first.shape == (1, 9)
    # the mask removes ceil-ish floor of the ratio: removal = L - int(L * (1 - ratio))
    assert int((first == 1).sum()) == 9 - int(9 * (1 - 0.6)) == 6

    # the mask is the declared identity-seeded rule, not an arbitrary draw
    expected = gen_random_mask(
        (3, 3),
        1,
        0.6,
        generator=torch.Generator(device="cpu").manual_seed(mask_seed("ep0", "cam0", "f0")),
    )
    assert torch.equal(first, expected)
    assert mask_seed("ep0", "cam0", "f0") != mask_seed("ep0", "cam0", "f1")
    assert mask_seed("ep0", "cam0", "f0") != mask_seed("ep0", "cam1", "f0")
    assert mask_seed("ep0", "cam0", "f0") != mask_seed("ep1", "cam0", "f0")

    # identity changes must change the mask (three independent identity axes)
    assert not torch.equal(
        first, pretrainer.generate_mask(_frame(values, frame_ids=("f1",)), camera_id="cam0")
    )
    assert not torch.equal(
        first, pretrainer.generate_mask(_frame(values, camera_id="cam1"), camera_id="cam1")
    )
    assert not torch.equal(
        first, pretrainer.generate_mask(_frame(values, episode_ids=("ep1",)), camera_id="cam0")
    )


def test_fcmae_mask_depends_on_support_and_never_on_pixels_future_or_truth():
    pretrainer = _pretrainer()
    torch.manual_seed(43)
    values = torch.rand(1, 3, 96, 96)
    frame = _frame(values)
    base = pretrainer.generate_mask(frame, camera_id="cam0")

    # pixel content cannot move the mask
    other_values = pretrainer.generate_mask(_frame(torch.rand(1, 3, 96, 96)), camera_id="cam0")
    assert torch.equal(base, other_values)

    # truth / future / evaluation fields in metadata cannot move the mask
    noisy_metadata = {
        "truth": {"object_pose": [1.0, 2.0]},
        "future_frame": "f999",
        "evaluation": {"success": True},
    }
    assert torch.equal(
        base, pretrainer.generate_mask(_frame(values, metadata=noisy_metadata), camera_id="cam0")
    )

    # a different camera's frame (a future frame) cannot move this camera's mask
    frames = {
        "cam0": _frame(values),
        "cam1": _frame(torch.rand(1, 3, 96, 96), camera_id="cam1", frame_ids=("f1",)),
    }
    out = pretrainer(frames=frames, camera_ids=("cam0", "cam1"))
    changed = dict(frames)
    changed["cam1"] = _frame(
        torch.rand(1, 3, 96, 96) * 100.0,
        camera_id="cam1",
        frame_ids=("f1",),
        metadata=noisy_metadata,
    )
    out_changed = pretrainer(frames=changed, camera_ids=("cam0", "cam1"))
    assert torch.equal(out.mask[0], out_changed.mask[0])
    assert torch.equal(out.mask[1], out_changed.mask[1])  # masks follow identity, not pixels

    # support changes the mask: a fully unsupported patch must be removed
    support = torch.ones(1, 96, 96, dtype=torch.bool)
    support[0, 32:64, 0:32] = False
    forced = pretrainer.generate_mask(_frame(values, valid_mask=support), camera_id="cam0")
    assert forced[0, 3] == 1  # patch (1, 0) has no support at all
    assert int((forced == 1).sum()) >= int((base == 1).sum())

    # the pretraining call reads no supervision: no targets/truth parameter exists
    parameters = inspect.signature(RawFCMAEPretrainer.forward).parameters
    assert set(parameters) == {"self", "frames", "camera_ids"}
    assert not any(token in str(parameters) for token in ("target", "truth", "future"))


# --------------------------------------------------------------------------------------
# steps 6: FCMAE loss, counts and gradients
# --------------------------------------------------------------------------------------


def _target_patch(mask_row: torch.Tensor, grid: int = 3) -> tuple[int, int]:
    index = int((mask_row == 1).nonzero()[0, 0])
    return divmod(index, grid)


def test_fcmae_loss_counts_only_valid_non_saturated_samples_and_gradients_follow():
    pretrainer = _pretrainer()
    torch.manual_seed(47)
    values = torch.rand(1, 3, 96, 96)
    base_frame = _frame(values)

    base = pretrainer(frames={"cam0": base_frame}, camera_ids=("cam0",))
    mask = base.mask[0]
    assert mask.shape == (1, 9)
    row, column = _target_patch(mask[0])
    rows = slice(row * 32, (row + 1) * 32)
    columns = slice(column * 32, (column + 1) * 32)
    blocked = 32 * 32 * 3
    contributing = float((mask[0] == 1).sum()) * 32 * 32 * 3
    assert int(base.valid_target_count) == int(contributing)

    # missing / bad samples leave the loss: the target count drops and their values do not
    # reach either the encoder input or the reconstruction target
    for name, kwargs in (
        ("valid_mask", {"valid_mask": _all_but(values.shape, rows, columns, default=True)}),
        ("bad_mask", {"bad_mask": _all_but(values.shape, rows, columns, default=False)}),
    ):
        blocked_frame = _frame(values, **kwargs)
        blocked_out = pretrainer(frames={"cam0": blocked_frame}, camera_ids=("cam0",))
        assert torch.equal(blocked_out.mask[0], mask), name
        assert int(blocked_out.valid_target_count) == int(contributing) - blocked
        assert int(blocked_out.saturated_count) == 0
        perturbed_values = values.clone()
        perturbed_values[:, :, rows, columns] = torch.rand(1, 3, 32, 32) * 1000.0
        perturbed = pretrainer(
            frames={"cam0": _frame(perturbed_values, **kwargs)}, camera_ids=("cam0",)
        )
        assert torch.equal(perturbed.loss, blocked_out.loss), name
        assert int(perturbed.valid_target_count) == int(blocked_out.valid_target_count)

        # the excluded samples contribute no gradient either: the two runs must produce
        # bit-identical gradients even though their hidden pixel values differ wildly
        grad_blocked = _pred_gradient(pretrainer, blocked_out.loss)
        grad_perturbed = _pred_gradient(pretrainer, perturbed.loss)
        assert torch.equal(grad_blocked, grad_perturbed), name
        assert torch.isfinite(grad_blocked).all() and float(grad_blocked.abs().max()) > 0.0

    # a saturated sample is reported separately and no longer contributes to the loss
    saturated_mask = _all_but(values.shape, rows, columns, default=False)
    saturated = pretrainer(
        frames={"cam0": _frame(values, saturated_mask=saturated_mask)}, camera_ids=("cam0",)
    )
    assert torch.equal(saturated.mask[0], mask)
    assert int(saturated.saturated_count) == blocked
    assert int(saturated.valid_target_count) == int(contributing) - blocked
    assert not torch.equal(saturated.loss, base.loss)

    # gradients flow to every stage of the pretraining path and stay finite
    base.loss.backward()
    for name in ("pred.weight", "decoder.0.dwconv.weight", "proj.weight"):
        parameter = dict(pretrainer.fcmae.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert float(dict(pretrainer.fcmae.named_parameters())["pred.weight"].grad.abs().max()) > 0.0
    encoder_stage = dict(pretrainer.fcmae.named_parameters())["encoder.stages.0.0.pwconv1.weight"]
    assert encoder_stage.grad is not None and torch.isfinite(encoder_stage.grad).all()


def _all_but(shape, rows: slice, columns: slice, *, default: bool | np.bool_) -> torch.Tensor:
    mask = torch.full((1, shape[-2], shape[-1]), bool(default), dtype=torch.bool)
    mask[:, rows, columns] = not default
    return mask


def _pred_gradient(pretrainer: RawFCMAEPretrainer, loss: torch.Tensor) -> torch.Tensor:
    """Gradient of ``fcmae.pred.weight`` for one loss, without accumulating state."""
    pretrainer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = dict(pretrainer.fcmae.named_parameters())["pred.weight"].grad
    assert gradient is not None
    return gradient.detach().clone()


def _assert_row_matches_single_row(
    batched: FeatureLevel, single: FeatureLevel, row: int, *, label: str
) -> None:
    """Row ``row`` of a batched level equals the single-row run of the same frame.

    Dispatcher-accepted comparison contract (2026-10-14 ruling): this is **not** a strict
    bitwise proof for feature values, and must not be cited as one.

    * ``valid`` and ``sensor_coordinates`` are asserted bitwise. Support is integer
      bookkeeping and the coordinate grid is analytic, so both are exactly equal.
    * ``values`` are asserted within ``1e-5`` of the largest magnitude. A CPU convolution
      selects a different kernel for a different batch size: measured on this backbone, the
      stem and stage 0 are bitwise equal, the divergence starts at the second convolution and
      is one float32 ULP (max abs 8.9e-8, relative ~1e-5). Kernel rounding is not a semantic
      difference; forcing bitwise equality would need MKLDNN disabled or a per-row forward,
      both of which change the configuration under test.
    * Batch-row independence of ``values`` *is* asserted bitwise in the batch-axis tests
      (replacing another row's pixels leaves this row bitwise identical).
    """
    assert torch.equal(batched.valid[row : row + 1], single.valid), label
    assert torch.equal(batched.sensor_coordinates[row : row + 1], single.sensor_coordinates), label
    reference = batched.values[row : row + 1].detach()
    candidate = single.values.detach()
    scale = max(float(reference.abs().max()), float(candidate.abs().max()), 1e-6)
    assert float((reference - candidate).abs().max()) <= 1e-5 * scale, label


def test_fcmae_multicamera_layout_loss_aggregation_and_heterogeneous_error():
    pretrainer = _pretrainer()
    torch.manual_seed(53)
    frame_a = _frame(torch.rand(1, 3, 96, 96), camera_id="a", frame_ids=("fa",))
    frame_b = _frame(torch.rand(1, 3, 96, 96), camera_id="b", frame_ids=("fb",))

    together = pretrainer(frames={"a": frame_a, "b": frame_b}, camera_ids=("a", "b"))
    assert together.reconstruction.shape == (2, 1, 3, 96, 96)
    assert together.mask.shape == (2, 1, 9)

    alone_a = pretrainer(frames={"a": frame_a}, camera_ids=("a",))
    alone_b = pretrainer(frames={"b": frame_b}, camera_ids=("b",))
    count_a = int(alone_a.valid_target_count)
    count_b = int(alone_b.valid_target_count)
    expected = (float(alone_a.loss.detach()) * count_a + float(alone_b.loss.detach()) * count_b) / (
        count_a + count_b
    )
    assert abs(float(together.loss.detach()) - expected) < 1e-6
    assert int(together.valid_target_count) == count_a + count_b
    assert torch.equal(together.mask[0], alone_a.mask[0])
    assert torch.equal(together.mask[1], alone_b.mask[0])

    with pytest.raises(ValueError, match="share"):
        pretrainer(
            frames={
                "a": frame_a,
                "c": _frame(torch.rand(1, 3, 64, 96), camera_id="c", frame_ids=("fc",)),
            },
            camera_ids=("a", "c"),
        )
    with pytest.raises(KeyError):
        pretrainer(frames={"a": frame_a}, camera_ids=("missing",))
    with pytest.raises(ValueError):
        pretrainer(frames={}, camera_ids=())


def test_fcmae_targets_follow_the_declared_calibration_and_stay_out_of_the_encoder():
    pretrainer = _pretrainer()
    torch.manual_seed(59)
    raw = torch.rand(1, 3, 96, 96) * 1000.0 + 100.0
    declared = pretrainer(
        frames={
            "cam0": _frame(
                raw, metadata={"calibration": {"black_level": 100.0, "white_level": 1100.0}}
            )
        },
        camera_ids=("cam0",),
    )
    explicit = pretrainer(frames={"cam0": _frame((raw - 100.0) / 1000.0)}, camera_ids=("cam0",))
    assert torch.equal(declared.loss, explicit.loss)
    assert torch.equal(declared.reconstruction, explicit.reconstruction)
    assert int(declared.valid_target_count) == int(explicit.valid_target_count)

    # the reconstruction is a training output only: it is a leaf-free tensor with gradient
    assert declared.reconstruction.grad_fn is not None
    assert declared.loss.requires_grad
    # and the observation encoder never exposes it
    encoder_outputs = _encoder(RAW3).encode(_bundle({"cam0": _frame(torch.rand(1, 3, 96, 96))}))
    assert not hasattr(encoder_outputs, "reconstruction")
    assert all("reconstruction" not in name for name in encoder_outputs.__dataclass_fields__)


def test_frame_transform_is_a_pure_record_of_padding_and_crop():
    frame = _frame(torch.zeros(1, 3, 62, 100), metadata={"camera": {"crop_offset": [7, 9]}})
    transform = frame_transform(frame, input_mode=RAW3)
    assert transform.padded_shape == (64, 128)
    assert transform.pad_after == (2, 28)
    assert transform.pad_before == (0, 0)
    assert transform.sensor_shape == (69, 109)
    assert transform.to_dict()["crop_offset"] == [7, 9]
    # the plan is stable and depends only on shape/mode/alignment
    assert frame_transform(frame, input_mode=RAW3) == transform
    assert frame_transform(frame, input_mode=RAW3, alignment=32).alignment == 32


# --------------------------------------------------------------------------------------
# steps 7: CUDA device consistency (T4 regression: CPU mask combined with CUDA support)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_pretraining_mask_is_device_consistent_and_forces_unsupported_patches():
    """The identity-seeded mask is drawn on CPU and must join CUDA support without a crash.

    On the pre-fix code ``torch.maximum(mask, unsupported)`` mixed a CPU mask with a CUDA
    support map and raised ``RuntimeError: Expected all tensors to be on the same device``
    on every CUDA pretraining step.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    pretrainer = _pretrainer().to(device)
    torch.manual_seed(71)
    values = torch.rand(1, 3, 96, 96)
    frame = _frame(values.to(device))

    out = pretrainer(frames={"cam0": frame}, camera_ids=("cam0",))
    assert out.mask.device == device
    assert out.loss.device == device
    assert out.reconstruction.device == device
    assert out.valid_target_count.device == device
    assert out.saturated_count.device == device
    masked_patches = int((out.mask[0] == 1).sum().item())
    assert int(out.valid_target_count.item()) == masked_patches * 32 * 32 * 3

    # one acquisition keeps one mask across devices: the CUDA mask equals the CPU mask
    cpu_out = _pretrainer()(frames={"cam0": _frame(values)}, camera_ids=("cam0",))
    assert torch.equal(out.mask.cpu(), cpu_out.mask)
    assert torch.equal(out.valid_target_count.cpu(), cpu_out.valid_target_count)

    # an unsupported patch is forced into the mask and contributes no target sample
    support = torch.ones(1, 96, 96, dtype=torch.bool, device=device)
    support[0, 32:64, 0:32] = False
    forced = pretrainer(
        frames={"cam0": _frame(values.to(device), valid_mask=support)}, camera_ids=("cam0",)
    )
    assert float(forced.mask[0, 0, 3].item()) == 1.0
    target_pixels = upsample_patch_mask(forced.mask[0], 32, (3, 3)).unsqueeze(1)
    expected = int((target_pixels * support.unsqueeze(1)).sum().item()) * 3
    assert int(forced.valid_target_count.item()) == expected

    # gradients reach CUDA parameters and stay finite
    out.loss.backward()
    gradient = dict(pretrainer.fcmae.named_parameters())["pred.weight"].grad
    assert gradient is not None and gradient.device == device
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().max().item()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_encoder_levels_keep_coordinates_on_the_feature_device():
    device = torch.device("cuda", torch.cuda.current_device())
    encoder = _encoder(RAW3).to(device)
    torch.manual_seed(73)
    frame = _frame(torch.rand(1, 3, 64, 96).to(device))
    with torch.no_grad():
        pyramid = encoder.encode(_bundle({"cam0": frame}))
    assert pyramid.transforms["cam0"].frame_shape == (64, 96)
    for level_id in ("stride8", "stride16"):
        level = pyramid.levels[level_id]
        assert level.values.device == device
        assert level.valid.device == device
        assert level.sensor_coordinates.device == device
        assert level.sensor_coordinates.dtype == torch.float32
    tokens = pyramid.tokens()
    assert tokens.values.device == device
    assert tokens.camera_index.device == device

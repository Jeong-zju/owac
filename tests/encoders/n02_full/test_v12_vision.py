"""N02 v1.2 vision front end tests: RAW encoder, SAVi measurement decoder, Perceiver readout.

These are real numerical checks, not import checks:

* the stride 4/8/16 encoder keeps the P0/P1 helpers (coordinates, padding, validity) and is
  exactly the P0-parity backbone restricted to stages 0..2;
* the decoder emits sensor-domain means with strictly positive variances, its saturation
  handling is a censored tail probability rather than an exact intensity, and the ``log σ²``
  term of the NLL is differentiable with respect to the decoder's variance head;
* the Perceiver attention port is compared against ``torch.nn.MultiheadAttention`` with the
  same weights, and the readout's attention tensors are normalized as documented;
* the three modules are differentiable end to end.

The Base parameter budget (~89M) is asserted without a forward pass so CPU tests stay cheap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from owac.core.encoder_contract_n02 import EmbodimentContext, MeasurementFrame
from owac.frameworks.n02_full.components.convnextv2 import DEFAULT_DEPTHS, DEFAULT_DIMS
from owac.frameworks.n02_full.decoder_v12 import (
    MASK_NAMES_V12,
    RawMeasurementDecoderV12,
    SaviPositionEmbedding,
    SpatialBroadcastDecoderV12,
    convert_to_fourier_features,
    measurement_nll,
    spatial_broadcast,
)
from owac.frameworks.n02_full.encoder_v12 import (
    LEVEL_IDS_V12,
    LEVEL_STRIDES_V12,
    RawEncoderV12,
    SpatialV12,
)
from owac.frameworks.n02_full.readout_v12 import (
    FourierPositionEncoding,
    N02ReadoutV12,
    PerceiverAttention,
    attend,
    build_linear_positions,
    generate_fourier_features,
)
from owac.frameworks.n02_full.sensor import clip_tail_log_prob, gaussian_nll
from owac.frameworks.n02_full.state import ObservationBundle

SMALL_DEPTHS = (1, 1, 1, 1)
SMALL_DIMS = (8, 16, 32, 64)
D_MODEL = 16


def make_frame(
    height: int,
    width: int,
    camera_id: str,
    *,
    batch: int = 1,
    scale: float = 3000.0,
    valid: bool = True,
    metadata: dict | None = None,
) -> MeasurementFrame:
    """One small RAW3 measurement frame in RoboTwin-like float units (never divided by 255)."""
    rng = np.random.default_rng(sum(ord(char) for char in camera_id))
    values = (rng.random((batch, 3, height, width)) * scale).astype("float32")
    return MeasurementFrame(
        values=values,
        valid_mask=np.full((batch, height, width), valid, dtype=bool),
        bad_mask=np.zeros((batch, height, width), dtype=bool),
        saturated_mask=np.zeros((batch, height, width), dtype=bool),
        episode_ids=tuple(f"episode{i}" for i in range(batch)),
        frame_ids=tuple(f"frame{i}" for i in range(batch)),
        camera_id=camera_id,
        metadata={"camera": {"crop_offset": [0, 0]}, **(metadata or {})},
        metadata_available={},
    )


def make_bundle(frames: dict[str, MeasurementFrame], *, batch: int = 1) -> ObservationBundle:
    return ObservationBundle(
        frames=frames,
        context=EmbodimentContext(),
        episode_ids=tuple(f"episode{i}" for i in range(batch)),
        anchor_ids=tuple(f"frame{i}" for i in range(batch)),
    )


def small_encoder(camera_ids=("top", "left")) -> RawEncoderV12:
    return RawEncoderV12(
        camera_ids=camera_ids,
        dims=SMALL_DIMS,
        depths=SMALL_DEPTHS,
        d_model=D_MODEL,
    )


def small_bundle(*, batch: int = 1, valid: bool = True) -> tuple[RawEncoderV12, SpatialV12]:
    encoder = small_encoder()
    bundle = make_bundle(
        {
            "top": make_frame(32, 64, "top", batch=batch, valid=valid),
            "left": make_frame(64, 32, "left", batch=batch, valid=valid),
        },
        batch=batch,
    )
    return encoder, encoder(bundle)


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


def test_encoder_returns_three_levels_per_camera_with_real_coordinates() -> None:
    encoder, spatial = small_bundle()
    assert spatial.level_ids == LEVEL_IDS_V12 == ("stride4", "stride8", "stride16")
    assert spatial.camera_ids == ("top", "left")
    assert set(spatial.levels) == {
        "top::stride4",
        "top::stride8",
        "top::stride16",
        "left::stride4",
        "left::stride8",
        "left::stride16",
    }
    assert spatial.level_dims == {key: D_MODEL for key in spatial.levels}
    assert encoder.spatial_dims == {level: D_MODEL for level in LEVEL_IDS_V12}

    grouped = spatial.per_camera()
    assert set(grouped) == {"top", "left"}
    top = grouped["top"]
    # 32x64 needs no padding (both multiples of the 32-cell alignment): 8x16 / 4x8 / 2x4.
    assert tuple(top["stride4"].values.shape) == (1, D_MODEL, 8, 16)
    assert tuple(top["stride8"].values.shape) == (1, D_MODEL, 4, 8)
    assert tuple(top["stride16"].values.shape) == (1, D_MODEL, 2, 4)
    assert isinstance(top["stride4"], type(top["stride8"]))

    # Coordinates are sensor pixel centres at the level stride: (i + 0.5) * stride.
    coords = top["stride4"].sensor_coordinates
    assert coords.dtype == torch.float32
    assert torch.allclose(coords[0, 0, 0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(coords[0, 1, 2], torch.tensor([6.0, 10.0]))
    assert torch.allclose(top["stride16"].sensor_coordinates[0, 1, 2], torch.tensor([24.0, 40.0]))
    assert bool(top["stride4"].valid.all())


def test_encoder_padding_cells_are_invalid_and_exactly_zero() -> None:
    encoder = small_encoder(("top",))
    # 20x24 frame -> padded to 32x32: the bottom/right padding must be invalid and zero.
    bundle = make_bundle({"top": make_frame(20, 24, "top")})
    spatial = encoder(bundle)
    assert spatial.transforms["top"].pad_after == (12, 8)
    level = spatial.levels["stride4"]
    valid = spatial.valid["stride4"]
    assert tuple(level.shape[-2:]) == (8, 8)
    assert tuple(valid.shape[-2:]) == (8, 8)
    # Row 5..7 and column 6..7 cover padding only.
    assert not bool(valid[0, 6:, :].any()) and not bool(valid[0, :, 6:].any())
    padded = level[0][:, ~valid[0]]
    assert torch.equal(padded, torch.zeros_like(padded))


def test_encoder_backbone_restriction_matches_full_multiscale() -> None:
    """Stages 0..2 of the v1.2 encoder are exactly the P0 backbone's own stages."""
    encoder = small_encoder(("top",))
    frame = make_frame(32, 48, "top")
    from owac.frameworks.n02_full.observation import _prepare_frame

    prepared = _prepare_frame(frame, "raw3", alignment=32, device=torch.device("cpu"))
    reference = encoder.backbone.forward_multiscale(prepared.values)
    mine = encoder._backbone_levels(prepared.values)
    for key in ("stage0_stride4", "stage1_stride8", "stage2_stride16"):
        assert torch.equal(reference[key], mine[key])


def test_encoder_base_configuration_has_official_parameter_budget() -> None:
    encoder = RawEncoderV12(camera_ids=("top",))
    backbone = sum(parameter.numel() for parameter in encoder.backbone.parameters())
    assert encoder.dims == tuple(DEFAULT_DIMS) and encoder.depths == tuple(DEFAULT_DEPTHS)
    # The public ConvNeXt V2-Base size (~89M) includes the 1000-class head; the head-less
    # encoder this project uses is 87,692,800 parameters. The three 1x1 projections add
    # (128 + 256 + 512) * 512 = 458,752.
    assert 87_600_000 < backbone < 87_800_000
    projected = encoder.parameter_count() - backbone
    assert projected == (128 + 256 + 512) * 512 + 3 * 512  # weights and biases
    assert encoder.spatial_dims == {level: 512 for level in LEVEL_IDS_V12}


def test_encoder_validity_follows_the_declared_mask() -> None:
    encoder, spatial = small_bundle(valid=False)
    assert not bool(spatial.valid["top::stride4"].any())
    assert torch.equal(
        spatial.levels["top::stride4"], torch.zeros_like(spatial.levels["top::stride4"])
    )


def test_encoder_rejects_unknown_camera_and_bad_modes() -> None:
    encoder = small_encoder(("top",))
    with pytest.raises(ValueError, match="not configured"):
        encoder(make_bundle({"wrist": make_frame(32, 32, "wrist")}))
    with pytest.raises(ValueError, match="unknown input_mode"):
        RawEncoderV12(camera_ids=("top",), input_mode="rgb3")
    with pytest.raises(ValueError, match="alignment"):
        RawEncoderV12(camera_ids=("top",), alignment=0)


def test_single_camera_pyramid_uses_plain_level_keys() -> None:
    encoder = small_encoder(("top",))
    spatial = encoder(make_bundle({"top": make_frame(32, 32, "top")}))
    assert set(spatial.levels) == set(LEVEL_IDS_V12)
    assert set(spatial.per_camera()["top"]) == set(LEVEL_IDS_V12)
    assert len(spatial.support_hash) == 64


# --------------------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------------------


def decoder_setup(**kwargs) -> tuple[RawMeasurementDecoderV12, SpatialV12, torch.Tensor]:
    _, spatial = small_bundle()
    slots = torch.randn(1, 4, 16, 32, generator=torch.Generator().manual_seed(0))
    decoder = RawMeasurementDecoderV12(
        in_dims=spatial.level_dims,
        slot_dim=32,
        n_slots=16,
        n_candidates=4,
        max_cameras=4,
        hidden=32,
        decoder_features=(16, 16, 16),
        num_fourier_bases=3,
        **kwargs,
    )
    return decoder, spatial, slots


def camera_queries(
    spatial: SpatialV12, *, per_camera: int = 6
) -> tuple[torch.Tensor, torch.Tensor]:
    coords, index = [], []
    for position, camera in enumerate(spatial.camera_ids):
        reference = spatial.per_camera()[camera]["stride4"].sensor_coordinates
        picked = reference.reshape(reference.shape[0], -1, 2)[:, :per_camera]
        coords.append(picked)
        index.append(torch.full((reference.shape[0], per_camera), position, dtype=torch.long))
    return torch.cat(coords, dim=1), torch.cat(index, dim=1)


def test_decoder_shapes_variance_and_masks() -> None:
    decoder, spatial, slots = decoder_setup()
    coords, camera_index = camera_queries(spatial)
    mean, variance, masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([3]),
        query_coords=coords,
        query_camera_index=camera_index,
    )
    batch, n_candidates, n_queries = 1, 4, coords.shape[1]
    assert tuple(mean.shape) == (batch, n_candidates, n_queries)
    assert tuple(variance.shape) == tuple(mean.shape)
    assert bool(torch.isfinite(mean).all()) and bool(torch.isfinite(variance).all())
    assert float(variance.min()) > 0.0
    for name in MASK_NAMES_V12:
        assert tuple(masks[name].shape) == tuple(mean.shape)
        assert float(masks[name].min()) >= 0.0 and float(masks[name].max()) <= 1.0
    assert masks["in_support"].dtype == torch.bool
    assert "saturated_censored" not in masks  # only declared with a clip level
    # Real queries inside the frame are in support; a far-away query is not.
    assert bool(masks["in_support"].all())
    far = torch.full_like(coords, 1e5)
    _, _, far_masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([3]),
        query_coords=far,
        query_camera_index=camera_index,
    )
    assert not bool(far_masks["in_support"].any())


def test_decoder_query_grid_maps_cell_centres_exactly() -> None:
    decoder, spatial, _ = decoder_setup()
    reference = spatial.per_camera()["top"]["stride4"].sensor_coordinates
    grid, inside = decoder._query_grid(spatial, 0, reference.reshape(1, -1, 2))
    assert bool(inside.all())
    height, width = reference.shape[1], reference.shape[2]
    rows = torch.arange(height, dtype=torch.float32)
    columns = torch.arange(width, dtype=torch.float32)
    expected_x = ((2.0 * columns + 1.0) / width - 1.0).repeat(height)
    expected_y = ((2.0 * rows + 1.0) / height - 1.0).repeat_interleave(width)
    assert torch.allclose(grid[0, :, 0], expected_x, atol=1e-6)
    assert torch.allclose(grid[0, :, 1], expected_y, atol=1e-6)


def test_decoder_log_variance_term_is_differentiable_and_positive() -> None:
    decoder, spatial, slots = decoder_setup()
    coords, camera_index = camera_queries(spatial)
    mean, variance, masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([0]),
        query_coords=coords,
        query_camera_index=camera_index,
    )
    target = mean.detach() + 2.0
    loss = measurement_nll(target, mean, variance, masks={"in_support": masks["in_support"]})
    manual = gaussian_nll(target, mean, variance).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    loss.backward()
    head = decoder.broadcast_decoder.target_readout.heads["log_variance"].out_proj
    assert head.bias.grad is not None and float(head.bias.grad.abs().sum()) > 0.0
    assert bool(torch.isfinite(head.bias.grad).all())
    # The log sigma^2 term bounds the variance: the proper score is minimised at
    # v = residual^2, so inflating the variance past the residual is strictly worse and an
    # unbounded variance is not a way to escape the residual.
    data_masks = {"in_support": masks["in_support"]}
    residual = 2.0
    target = mean.detach() + residual
    at_optimum = measurement_nll(
        target, mean, torch.full_like(variance, residual**2), masks=data_masks
    )
    inflated = measurement_nll(
        target, mean, torch.full_like(variance, (10.0 * residual) ** 2), masks=data_masks
    )
    assert float(inflated) > float(at_optimum)


def test_decoder_saturation_is_scored_as_a_censored_tail() -> None:
    decoder, spatial, slots = decoder_setup(saturation_level=500.0)
    coords, camera_index = camera_queries(spatial)
    mean, variance, masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([0]),
        query_coords=coords,
        query_camera_index=camera_index,
    )
    # Force one query column to be saturated: a mean above the declared clip level.
    saturated = torch.zeros_like(mean, dtype=torch.bool)
    saturated[:, :, 0] = True
    mean = torch.where(saturated, torch.full_like(mean, 900.0), mean)
    assert bool(masks["in_support"].all())
    data_masks = {"in_support": masks["in_support"], "saturated": saturated}
    target = torch.full_like(mean, 900.0)
    tail = -clip_tail_log_prob(mean[:, :, 0], variance[:, :, 0], level=500.0, side="top")
    censored = measurement_nll(
        target, mean, variance, masks=data_masks, saturation_level=500.0, reduction="none"
    )
    # A censored saturated entry is scored by the clip tail probability of the *declared*
    # level, not by the clipped value treated as an exact intensity.
    assert torch.allclose(censored[:, :, 0], tail, atol=1e-6)
    assert float(censored[:, :, 0].mean()) < 20.0  # -log P(y >= 500) is O(1)..O(100)
    # Without a declared level the same entry is explicitly not scored.
    unscored = measurement_nll(
        target, mean, variance, masks=data_masks, saturation_level=None, reduction="none"
    )
    assert float(unscored[:, :, 0].abs().max()) == 0.0
    assert torch.allclose(
        unscored[:, :, 1],
        gaussian_nll(target[:, :, 1], mean[:, :, 1], variance[:, :, 1]),
        atol=1e-6,
    )


def test_decoder_channel_selection_and_validation() -> None:
    decoder, spatial, slots = decoder_setup(measurement_channels=3, channel_reduction="select")
    coords, camera_index = camera_queries(spatial)
    channel = torch.zeros(coords.shape[:2], dtype=torch.long)
    mean, variance, _ = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([0]),
        query_coords=coords,
        query_camera_index=camera_index,
        query_channel=channel,
    )
    assert tuple(mean.shape) == (1, 4, coords.shape[1])
    assert float(variance.min()) > 0.0
    with pytest.raises(ValueError, match="query_channel"):
        decoder(
            features=spatial,
            slots=slots,
            frame_index=torch.tensor([0]),
            query_coords=coords,
            query_camera_index=camera_index,
        )
    with pytest.raises(ValueError, match="candidates"):
        decoder(
            features=spatial,
            slots=slots[:, :3],
            frame_index=torch.tensor([0]),
            query_coords=coords,
            query_camera_index=camera_index,
            query_channel=channel,
        )
    with pytest.raises(ValueError, match="non-negative"):
        decoder(
            features=spatial,
            slots=slots,
            frame_index=torch.tensor([-1]),
            query_coords=coords,
            query_camera_index=camera_index,
            query_channel=channel,
        )


def test_decoder_output_is_a_raw_prediction_in_the_same_units() -> None:
    decoder, spatial, slots = decoder_setup(units="robotwin_raw")
    coords, camera_index = camera_queries(spatial)
    mean, variance, masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([1]),
        query_coords=coords,
        query_camera_index=camera_index,
    )
    prediction = decoder.to_raw_prediction(mean, variance, masks)
    assert prediction.units == "robotwin_raw"
    assert torch.equal(prediction.mean, mean)
    assert torch.equal(prediction.variance, variance)
    assert set(prediction.variance_components) == {"sensor", "model", "total"}
    assert bool((prediction.variance > 0).all())


def test_savi_broadcast_pieces_match_the_official_formulas() -> None:
    x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    broadcast = spatial_broadcast(x, (2, 3))
    assert tuple(broadcast.shape) == (1, 2, 3, 4)
    assert torch.equal(broadcast[0, 1, 2], x[0])
    features = convert_to_fourier_features(torch.tensor([[0.5, 1.0]]), 2)
    # [sin, sin(+pi/2)] over the two frequency bases of both dimensions.
    assert tuple(features.shape) == (1, 8)
    torch.manual_seed(0)
    embedding = SaviPositionEmbedding(6, embedding_type="fourier", num_fourier_bases=2)
    grid = torch.zeros(1, 4, 4, 6)
    assert tuple(embedding(grid).shape) == (1, 4, 4, 6)
    decoder = SpatialBroadcastDecoderV12(
        6, {"mean": 2, "log_variance": 2, "masks": 4}, hidden=8, decoder_features=(8,)
    )
    # One 2x transposed-conv layer: broadcast (2, 2) -> decoded (4, 4) fields.
    fields, alphas = decoder(torch.randn(2, 3, 6), (2, 2))
    assert tuple(fields["mean"].shape) == (2, 2, 4, 4)
    assert tuple(alphas.shape) == (2, 3, 4, 4)
    assert torch.allclose(alphas.sum(dim=1), torch.ones(2, 4, 4), atol=1e-6)


# --------------------------------------------------------------------------------------
# readout
# --------------------------------------------------------------------------------------


def readout_setup(**kwargs) -> tuple[N02ReadoutV12, SpatialV12, torch.Tensor, torch.Tensor]:
    _, spatial = small_bundle()
    configuration = {
        "spatial_dims": spatial.level_dims,
        "latent_dim": 32,
        "n_latents": 16,
        "n_queries": 16,
        "out_dim": 48,
        "num_heads": 4,
        "depth": 2,
        "state_dim": 8,
    }
    configuration.update(kwargs)
    readout = N02ReadoutV12(**configuration)
    state = torch.randn(1, 4, 16, 8, generator=torch.Generator().manual_seed(1))
    support = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    return readout, spatial, state, support


def test_readout_tokens_attention_and_aux() -> None:
    readout, spatial, state, support = readout_setup()
    tokens, aux = readout(features=spatial, state_tokens=state, support=support)
    n_cameras = len(spatial.camera_ids)
    assert n_cameras * readout.n_queries == tokens.shape[1]
    assert tuple(tokens.shape) == (1, n_cameras * 16, 48)  # n_queries=16 in this fixture
    assert set(aux) >= {
        "attention",
        "encoder_attention",
        "candidate_attention",
        "latents",
        "token_valid",
        "support",
    }
    assert tuple(aux["latents"].shape) == (1, readout.n_latents, readout.latent_dim)
    # Documented normalisation: softmax over latents / over input tokens (rows sum to 1).
    assert torch.allclose(aux["attention"].sum(dim=-1), torch.ones(1, n_cameras * 16), atol=1e-5)
    assert torch.allclose(aux["encoder_attention"].sum(dim=-1), torch.ones(1, 16), atol=1e-5)
    assert torch.allclose(aux["candidate_attention"].sum(dim=-1), torch.ones(1), atol=1e-5)
    assert bool((aux["candidate_attention"] >= 0).all())


def test_readout_matches_torch_multihead_attention() -> None:
    """One necessary numerical comparison for the ported Perceiver attention."""
    torch.manual_seed(7)
    mine = PerceiverAttention(
        qk_channels=32, v_channels=32, output_channels=32, num_heads=4, init_scale=1.0
    )
    reference = torch.nn.MultiheadAttention(32, 4, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.cat([mine._q.weight, mine._k.weight, mine._v.weight], dim=0)
        )
        reference.in_proj_bias.copy_(torch.cat([mine._q.bias, mine._k.bias, mine._v.bias]))
        reference.out_proj.weight.copy_(mine._out.weight)
        reference.out_proj.bias.copy_(mine._out.bias)
    query = torch.randn(2, 5, 32)
    key_value = torch.randn(2, 7, 32)
    output, weights = mine(query, key_value)
    expected, expected_weights = reference(
        query, key_value, key_value, need_weights=True, average_attn_weights=True
    )
    assert float((output - expected).abs().max()) < 1e-5
    assert float((weights.mean(dim=1) - expected_weights).abs().max()) < 1e-6


def test_perceiver_attention_mask_and_bias_semantics() -> None:
    query = torch.randn(1, 2, 4, 4)  # (B, T, H, D)
    key_value = torch.randn(1, 3, 4, 4)
    mask = torch.tensor([[[True, False, True], [False, False, False]]])
    output, weights = attend(query, key_value, key_value, attention_mask=mask)
    assert float(weights[0, :, 0, 1].abs().max()) == 0.0
    assert float(output[0, 1].abs().max()) == 0.0  # fully masked query row is wiped
    bias = torch.tensor([[0.0, 10.0, 0.0]])
    _, biased = attend(query, key_value, key_value, attention_bias=bias)
    assert float(biased[0, :, 0, 1].min()) > 0.5


def test_position_encoding_follows_the_declared_device_and_dtype() -> None:
    """The decoder query grid must be built where the projection lives (CUDA regression)."""
    grid = build_linear_positions((2, 2), device="cpu", dtype=torch.float64)
    assert grid.dtype == torch.float64
    encoding = FourierPositionEncoding(index_dims=(32, 32), num_bands=4)
    features = encoding(1, device="cpu", dtype=torch.float64)
    assert features.dtype == torch.float64
    readout, spatial, state, support = readout_setup()
    query = readout.decoder.decoder_query(2)
    assert query.device == readout.decoder.final_layer.weight.device
    assert query.dtype == readout.decoder.final_layer.weight.dtype
    assert tuple(query.shape) == (2, 16, 32)
    tokens, _ = readout(features=spatial, state_tokens=state, support=support)
    assert bool(torch.isfinite(tokens).all())


def test_perceiver_position_encoding_is_official_fourier() -> None:
    grid = build_linear_positions((2, 3), (-1.0, 1.0))
    assert tuple(grid.shape) == (2, 3, 2)
    assert float(grid[1, 2, 0]) == 1.0 and float(grid[1, 2, 1]) == 1.0
    features = generate_fourier_features(
        torch.tensor([[-1.0, 1.0], [1.0, 1.0]]), num_bands=4, max_resolution=(16, 16)
    )
    # concat_pos=True adds the 2 raw coordinates to sin/cos of 2 dims x 4 bands.
    assert tuple(features.shape) == (2, 2 + 2 * 2 * 4)


def test_readout_masked_tokens_receive_no_attention_mass() -> None:
    readout, spatial, state, support = readout_setup()
    tokens, aux = readout(features=spatial, state_tokens=state, support=support)
    mask = aux["token_valid"].clone()
    mask[:, :100] = False
    _, masked_aux = readout(
        features=spatial, state_tokens=state, support=support, masks={"input_valid": mask}
    )
    mass = masked_aux["encoder_attention"].sum(dim=1)[0]
    assert float(mass[:100].abs().max()) == 0.0
    assert float(mass[100:].min()) > 0.0
    assert tuple(tokens.shape) == tuple(
        readout(features=spatial, state_tokens=state, support=support)[0].shape
    )


def test_readout_support_weight_moves_candidate_attention() -> None:
    torch.manual_seed(11)
    readout, spatial, state, _ = readout_setup()
    _, aux_a = readout(
        features=spatial, state_tokens=state, support=torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    )
    _, aux_b = readout(
        features=spatial, state_tokens=state, support=torch.tensor([[0.1, 0.1, 0.1, 0.7]])
    )
    assert int(aux_a["candidate_attention"].argmax()) == 0
    assert int(aux_b["candidate_attention"].argmax()) == 3


def test_readout_depth_and_query_grid_configuration() -> None:
    readout, spatial, state, support = readout_setup(depth=4, n_queries=26)
    assert readout.query_grid == (26,)  # 26 is not a square: official 1-D index grid
    tokens, aux = readout(features=spatial, state_tokens=state, support=support)
    assert tuple(tokens.shape) == (1, 2 * 26, 48)
    assert len(readout.encoder.self_attends) == 1
    assert readout.encoder.num_blocks == 4
    assert tuple(aux["attention"].shape) == (1, 52, 16)
    with pytest.raises(ValueError, match="state_tokens"):
        readout(features=spatial, state_tokens=torch.randn(1, 4, 16, 7), support=support)


# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------


def test_three_modules_are_differentiable_end_to_end() -> None:
    encoder, spatial = small_bundle()
    decoder, _, slots = decoder_setup()
    readout = N02ReadoutV12(
        spatial_dims=spatial.level_dims,
        latent_dim=32,
        n_latents=8,
        n_queries=8,
        out_dim=16,
        num_heads=4,
        depth=1,
        state_dim=32,
    )
    slots = slots.detach().requires_grad_(True)
    coords, camera_index = camera_queries(spatial, per_camera=4)
    mean, variance, masks = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([2]),
        query_coords=coords,
        query_camera_index=camera_index,
    )
    nll = measurement_nll(
        mean.detach() + 1.0, mean, variance, masks={"in_support": masks["in_support"]}
    )
    support = torch.softmax(torch.randn(1, 4), dim=-1)
    tokens, _ = readout(features=spatial, state_tokens=slots, support=support)
    loss = nll + tokens.pow(2).mean()
    loss.backward()
    assert bool(torch.isfinite(loss))
    for name, module in (("encoder", encoder), ("decoder", decoder), ("readout", readout)):
        grads = [
            float(parameter.grad.abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert grads, f"{name} received no gradients"
        assert all(math.isfinite(value) for value in grads), f"{name} produced non-finite grads"
        assert max(grads) > 0.0, f"{name} received only zero gradients"
    assert slots.grad is not None and float(slots.grad.abs().sum()) > 0.0
    # The encoder's own gradients come from the spatial tokens that entered both branches.
    encoder_grads = [
        float(parameter.grad.abs().sum())
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    assert max(encoder_grads) > 0.0


def test_decoder_grid_stride_is_declared_per_level() -> None:
    assert LEVEL_STRIDES_V12 == {"stride4": 4, "stride8": 8, "stride16": 16}
    _, spatial = small_bundle()
    for key, level in spatial.levels.items():
        stride = LEVEL_STRIDES_V12[key.rsplit("::", 1)[-1]]
        coords = spatial.coords[key]
        assert torch.allclose(
            coords[0, 0, 0] / torch.tensor(float(stride)), torch.tensor([0.5, 0.5]), atol=1e-6
        )
        assert level.dtype == torch.float32


def test_zero_support_candidate_is_completely_masked_from_attention():
    readout, spatial, state, _ = readout_setup()
    _, aux = readout(
        features=spatial, state_tokens=state, support=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    assert not aux["candidate_attention"][0, 1:].any()


def test_declared_initial_variance_uses_inverse_softplus():
    decoder, spatial, slots = decoder_setup(init_variance=2.0)
    head = decoder.broadcast_decoder.target_readout.heads["log_variance"].out_proj
    torch.nn.init.zeros_(head.weight)
    coords, cameras = camera_queries(spatial)
    _, variance, _ = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([3]),
        query_coords=coords,
        query_camera_index=cameras,
    )
    torch.testing.assert_close(variance, torch.full_like(variance, 2.0), atol=1e-5, rtol=0)


def test_weighted_measurement_masks_nan_before_arithmetic_and_backward():
    from owac.frameworks.n02_full.checks_v12 import N02V12Measurement

    mean = torch.tensor([[[0.0, float("nan")]]], requires_grad=True)
    variance = torch.tensor([[[1.0, float("nan")]]], requires_grad=True)
    energy = N02V12Measurement().energy(
        mean=mean,
        variance=variance,
        masks=torch.tensor([[True, False]]),
        target=torch.tensor([[0.0, float("nan")]]),
        weights=torch.ones(1, 2),
    )
    torch.testing.assert_close(
        energy, torch.tensor([[0.5 * np.log(2 * np.pi)]], dtype=energy.dtype)
    )
    energy.sum().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(variance.grad).all()
    with pytest.raises(ValueError, match="weights"):
        N02V12Measurement().energy(
            mean=torch.zeros(1, 1, 2),
            variance=None,
            masks=torch.ones(1, 2),
            target=torch.zeros(1, 2),
            weights=-torch.ones(1, 2),
        )


def test_grouped_camera_backbone_preserves_spatial_values_and_gradients():
    encoder, spatial = small_bundle()
    frames = {camera: make_frame(32, 64, camera) for camera in spatial.camera_ids}
    observation = make_bundle(frames)
    encoder.batch_cameras = False
    sequential = encoder(observation)
    encoder.batch_cameras = True
    grouped = encoder(observation)
    assert grouped.support_hash == sequential.support_hash
    for key in sequential.levels:
        torch.testing.assert_close(
            grouped.levels[key], sequential.levels[key], atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(grouped.coords[key], sequential.coords[key], atol=0, rtol=0)
    sum(value.square().mean() for value in grouped.levels.values()).backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in encoder.parameters())


def test_camera_query_grouping_preserves_fields_and_slot_gradients():
    decoder, spatial, slots = decoder_setup(channel_reduction="select")
    coords, cameras = camera_queries(spatial)
    channels = torch.arange(coords.shape[1]).remainder(3).unsqueeze(0)
    slots.requires_grad_()
    reference = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([3]),
        query_coords=coords,
        query_camera_index=cameras,
        query_channel=channels,
    )
    ref_grad = torch.autograd.grad(reference[0].sum() + reference[1].sum(), slots)[0]
    decoder.group_camera_queries = True
    grouped = decoder(
        features=spatial,
        slots=slots,
        frame_index=torch.tensor([3]),
        query_coords=coords,
        query_camera_index=cameras,
        query_channel=channels,
    )
    grad = torch.autograd.grad(grouped[0].sum() + grouped[1].sum(), slots)[0]
    for a, b in zip(reference[:2], grouped[:2], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    for key in reference[2]:
        torch.testing.assert_close(reference[2][key], grouped[2][key], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-5, atol=1e-6)

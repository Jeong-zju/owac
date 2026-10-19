"""Behavioural tests for the P2 DBSR feature adapter (``n01_full/adapters.py``).

These tests are written to fail a wrong implementation, not to restate it:

* the pooled token mask is checked against an explicitly invalid quadrant and a
  fully invalid frame, and the invalid pixels' *values* are replaced with huge
  garbage to prove they never enter a token (a mask-blind average would move);
* per-camera tokens are compared with an independently driven single-camera
  adapter, so a shared or reordered cache cannot pass;
* the readout parameter accounting is compared with hand-computed counts and the
  absence of any ``2048 -> 2048`` projection is asserted directly;
* the image tower is a stub whose ``embed_image`` raises and counts calls, so a
  hidden image-path call cannot pass, and the ten-step flow solve is checked to
  leave both the encoder counter and the adapter arrival counters untouched.

The stub PI0 surface is an interface replica for plumbing only: it has none of
pi0.5's structure and no claim about real weights is made here.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from owac.backends.rawvla_features import ACTION_PROTOCOLS, RawVLAFeaturePolicy
from owac.baselines.burst_features import BurstFeatureConfig, BurstFeatureEncoder
from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
from owac.frameworks.n01_full.adapters import (
    P3_SUPPORT_ENABLED,
    DbsrAdapterConfig,
    DbsrFeatureAdapter,
)

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb")


def tiny_config(**overrides) -> DbsrAdapterConfig:
    """4-wide stem, one residual block: the same code path at CPU-test scale."""
    payload = {
        "stem_width": 4,
        "residual_blocks": 1,
        "feature_width": 8,
        "projection_width": 4,
        "offset_width": 8,
        "max_history": 4,
        "pool_grid": 2,
        "tokens_per_camera": 4,
        "readout_width": 16,
    }
    payload.update(overrides)
    return DbsrAdapterConfig(**payload)


def make_frame(
    values,
    *,
    camera="base_0_rgb",
    frame_ids=None,
    episode_ids=None,
    valid=None,
    timestamp=None,
    new_measurement=None,
):
    batch, channels, height, width = values.shape
    if valid is None:
        valid = torch.ones(batch, channels, height, width, dtype=torch.bool)
    return MeasurementFrame(
        values=values,
        valid_mask=valid,
        frame_ids=None if frame_ids is None else torch.as_tensor(frame_ids),
        episode_ids=episode_ids,
        camera_id=camera,
        timestamp=None if timestamp is None else torch.as_tensor(timestamp),
        new_measurement=new_measurement,
    )


def budget(adapter: DbsrFeatureAdapter) -> ComputeBudget:
    return ComputeBudget(max_updates=1, max_history=adapter.config.max_history)


def feed(adapter: DbsrFeatureAdapter, state, camera: str, count: int, **kwargs):
    output = None
    generator = torch.Generator().manual_seed(7)
    for step in range(count):
        values = torch.randn(1, adapter.config.input_channels, 8, 8, generator=generator)
        output = adapter.step(
            make_frame(values, camera=camera, frame_ids=[step], **kwargs),
            None,
            state,
            budget(adapter),
        )
        state = output.state
    return state, output


# ---------------------------------------------------------------------------------------
# pooling mask semantics (including a fully invalid cell)
# ---------------------------------------------------------------------------------------


def test_masked_pooling_returns_false_mask_and_zero_for_fully_invalid_cells():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.eval()
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    valid = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    valid[:, :, 4:, 4:] = True  # only the bottom-right 4x4 quadrant is measured
    with torch.no_grad():
        output = adapter.step(
            make_frame(torch.randn(1, 3, 8, 8), camera="a", frame_ids=[0], valid=valid),
            None,
            state,
            budget(adapter),
        )
        tokens = adapter.readout(output)
    # pool_grid=2 over 8x8: one 4x4 cell per token, row-major.
    assert tokens.mask[0].tolist() == [False, False, False, True]
    assert float(tokens.values[0, :3].abs().sum()) == 0.0
    assert float(tokens.values[0, 3].abs().sum()) > 0.0


def test_invalid_pixels_never_enter_a_token_even_with_garbage_values():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.eval()
    valid = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    valid[:, :, 4:, 4:] = True
    clean = torch.full((1, 3, 8, 8), 0.25)
    garbage = clean.clone()
    garbage[:, :, :4, :4] = 1e6  # invisible if the mask really gates the pooling

    def tokens_for(values):
        state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
        with torch.no_grad():
            output = adapter.step(
                make_frame(values, camera="a", frame_ids=[0], valid=valid),
                None,
                state,
                budget(adapter),
            )
            return adapter.readout(output)

    clean_tokens = tokens_for(clean)
    garbage_tokens = tokens_for(garbage)
    assert torch.equal(clean_tokens.values, garbage_tokens.values)
    assert torch.equal(clean_tokens.mask, garbage_tokens.mask)


def test_fully_invalid_frame_yields_all_false_mask_and_zero_values():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.eval()
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    valid = torch.zeros(1, 3, 8, 8, dtype=torch.bool)
    with torch.no_grad():
        output = adapter.step(
            make_frame(torch.randn(1, 3, 8, 8), camera="a", frame_ids=[0], valid=valid),
            None,
            state,
            budget(adapter),
        )
        tokens = adapter.readout(output)
    assert tokens.mask[0].tolist() == [False, False, False, False]
    assert float(tokens.values.abs().sum()) == 0.0


# ---------------------------------------------------------------------------------------
# camera order and per-camera independence
# ---------------------------------------------------------------------------------------


def test_camera_order_is_config_order_not_arrival_order():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.eval()
    frames = {
        camera: [
            torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(seed + step))
            for step in range(2)
        ]
        for camera, seed in (("a", 1), ("b", 2), ("c", 3))
    }

    def drive(camera_order, order):
        state = adapter.init_state(1, {"camera_ids": list(camera_order), "shape": (8, 8)}, "cpu")
        with torch.no_grad():
            for camera in order:
                output = None
                for step, values in enumerate(frames[camera]):
                    output = adapter.step(
                        make_frame(values, camera=camera, frame_ids=[step]),
                        None,
                        state,
                        budget(adapter),
                    )
                    state = output.state
            return adapter.readout(output)

    multi = drive(("a", "b", "c"), ("c", "a"))
    assert multi.camera_ids == ["a", "c"]
    single = drive(("a",), ("a",))
    assert torch.equal(multi.values[:, :4], single.values)
    multi_all = drive(("a", "b", "c"), ("c", "b", "a"))
    assert multi_all.camera_ids == ["a", "b", "c"]
    b_reference = drive(("b",), ("b",))
    assert torch.equal(multi_all.values[:, 4:8], b_reference.values)


# ---------------------------------------------------------------------------------------
# no hidden readout projection / parameter accounting
# ---------------------------------------------------------------------------------------


def test_parameter_accounting_has_one_512_to_2048_projection_and_no_2048_square():
    adapter = DbsrFeatureAdapter(DbsrAdapterConfig())  # full DBSR: 64 / 9 / 512
    report = adapter.parameter_report()
    # Hand-computed full DBSR counts (stem 3->64, 9 blocks of two 3x3 convs, 64->512 out).
    assert report["backbone"]["total"] == 961_920
    # MaskedDbsrFusion: 512->32 1x1, offset net, weight predictor (128->64, ResBlock, 64->512).
    assert report["fusion"]["total"] == 534_560
    assert report["total"] == 1_496_480

    post = adapter.build_postprocessor(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    assert (post.config.in_dim, post.config.width, post.config.tokens_per_camera) == (
        512,
        2048,
        256,
    )
    assert (post.in_proj.in_features, post.in_proj.out_features) == (512, 2048)
    expected = (
        512 * 2048  # in_proj, no bias because LayerNorm follows
        + 2 * 2048  # LayerNorm
        + 3  # per-camera gain
        + 3 * 2048  # camera embedding
        + 256 * 2048  # spatial index branch (spatial_mode="auto")
        + (16 * 2048 + 2048)  # spatial Fourier branch
    )
    assert post.parameter_report()["total"] == expected == 1_617_923
    for name, parameter in list(adapter.named_parameters()) + list(post.named_parameters()):
        assert tuple(parameter.shape) != (2048, 2048), name
    for name, module in list(adapter.named_modules()) + list(post.named_modules()):
        if isinstance(module, nn.Linear):
            assert not (module.in_features == 2048 and module.out_features == 2048), name

    # The deployed token contract: 256 tokens of 512 dimensions per camera.
    state = adapter.init_state(1, {"camera_ids": ["base_0_rgb"], "shape": (16, 16)}, "cpu")
    with torch.no_grad():
        output = adapter.step(
            make_frame(torch.randn(1, 3, 16, 16), camera="base_0_rgb", frame_ids=[0]),
            None,
            state,
            budget(adapter),
        )
        tokens = adapter.readout(output)
    assert tuple(tokens.values.shape) == (1, 256, 512)
    assert tuple(tokens.mask.shape) == (1, 256)
    assert bool(tokens.mask.all())


# ---------------------------------------------------------------------------------------
# the image tower never runs; the prefix comes from the visual tokens
# ---------------------------------------------------------------------------------------


class _StubVLM(nn.Module):
    def __init__(self, width: int, vocab: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.embed_image_calls = 0

    def embed_language_tokens(self, tokens):
        return self.embed(tokens)

    def embed_image(self, image):
        self.embed_image_calls += 1
        raise AssertionError("the image tower must never run on the raw_adapter path")

    def forward_prefix(
        self, *, inputs_embeds, attention_mask, position_ids, past_key_values, use_cache=True
    ):
        return SimpleNamespace(past_key_values=SimpleNamespace(prefix_embs=inputs_embeds))


class _StubActionHead(nn.Module):
    def __init__(self, width: int, action_dim: int = 4, horizon: int = 3) -> None:
        super().__init__()
        self.action_in = nn.Linear(action_dim, width)
        self.hidden = nn.Linear(width, width)
        self.out = nn.Linear(width, action_dim)
        self.horizon = horizon

    def sample_action_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def prepare_flow_targets(self, actions, noise=None, time=None):
        time = torch.full((actions.shape[0],), 0.5, device=actions.device) if time is None else time
        noise = torch.zeros_like(actions) if noise is None else noise
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        return x_t, noise - actions, noise, time

    def embed_action_suffix(self, state, x_t, timestep):
        embs = self.action_in(x_t)
        pad = torch.ones(embs.shape[:2], dtype=torch.bool, device=embs.device)
        att = torch.zeros_like(pad)
        att[:, :1] = True
        return embs, pad, att, None

    def decode_action_velocity(self, hidden):
        return self.out(hidden)


class _StubPolicy(nn.Module):
    """Interface replica of the PI0/PI05 surface used by ``RawVLAFeaturePolicy``."""

    def __init__(self, width: int, action_dim: int = 4, horizon: int = 3) -> None:
        super().__init__()
        self.vlm = _StubVLM(width)
        self.action_head = _StubActionHead(width, action_dim, horizon)
        self.action_dim = action_dim
        self.action_horizon = horizon
        self.max_state_dim = 4
        self.num_inference_steps = 10
        self.state_pad_value = 0.0
        self.discrete_state_input = False

    def forward_with_action_expert(
        self, *, prefix_embs, suffix_embs, attention_mask, position_ids, adarms_cond
    ):
        return self.action_head.hidden(suffix_embs) + prefix_embs.mean(1, keepdim=True)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        context = past_key_values.prefix_embs.mean(1, keepdim=True)
        hidden = self.action_head.hidden(self.action_head.action_in(x_t)) + context
        return self.action_head.decode_action_velocity(hidden)


def test_prefix_uses_visual_tokens_and_the_image_tower_is_never_called():
    torch.manual_seed(3)
    adapter = DbsrFeatureAdapter(tiny_config(input_mode="rawvla_rgb3"))
    post = adapter.build_postprocessor(CAMERAS)
    policy = _StubPolicy(width=adapter.config.readout_width)
    bridge = RawVLAFeaturePolicy(
        policy, encoder=adapter, postprocessor=post, action_protocol=ACTION_PROTOCOLS["libero"]
    )
    bridge.freeze_backend_parameters()

    state = adapter.init_state(2, {"camera_ids": list(CAMERAS), "shape": (8, 8)}, "cpu")
    generator = torch.Generator().manual_seed(5)
    output = None
    for step in range(3):
        for camera in CAMERAS:
            values = torch.randn(2, 3, 8, 8, generator=generator)
            frame = make_frame(
                values,
                camera=camera,
                frame_ids=[step, step],
                episode_ids=["e0", "e1"],
            )
            output = adapter.step(frame, None, state, budget(adapter))
            state = output.state
    tokens = adapter.readout(output)
    observation = SimpleNamespace(
        state=torch.zeros(2, 1, 4),
        tokenized_prompt=torch.randint(1, 32, (2, 5), generator=torch.Generator().manual_seed(4)),
        tokenized_prompt_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    actions = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(6))

    adapter.train()
    post.train()
    result = bridge.forward_from_observation_with_features(
        observation, tokens, actions, return_debug=True
    )
    loss = result["action_loss"]
    loss.backward()
    assert policy.vlm.embed_image_calls == 0
    assert result["prefix_source"] == "visual_tokens"
    assert result["mask_helper_source"]
    assert list(result["prefix_embs"][:, : 4 * len(CAMERAS)].shape) == [
        2,
        4 * len(CAMERAS),
        adapter.config.readout_width,
    ]
    assert torch.equal(result["pad_masks"][:, : 4 * len(CAMERAS)], tokens.mask)
    assert float(result["v_t"].detach().abs().sum()) > 0
    adapter_grad = sum(float(p.grad.norm()) for p in adapter.parameters() if p.grad is not None)
    assert adapter_grad > 0
    post_grad = sum(float(p.grad.norm()) for p in post.parameters() if p.grad is not None)
    assert post_grad > 0
    assert all(p.grad is None for p in policy.parameters())

    # One readout for the observation, then two ten-step solves reusing its prefix.
    bridge.encode_visual_tokens(output)
    calls_before = bridge.stats["encoder_calls"]
    steps_before = bridge.stats["denoise_steps"]
    arrivals_before = int(sum(int(value.sum()) for value in state.arrivals.values()))
    frozen = replace(tokens, values=tokens.values.detach(), positions=tokens.positions.detach())
    noise = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(8))
    first = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    second = bridge.sample_actions_with_features(
        observation, frozen, noise=noise.clone(), num_steps=10
    )
    assert bridge.stats["encoder_calls"] == calls_before
    assert bridge.stats["denoise_steps"] - steps_before == 20
    assert int(sum(int(value.sum()) for value in state.arrivals.values())) == arrivals_before
    assert float((first - second).abs().max()) == 0.0
    assert policy.vlm.embed_image_calls == 0


# ---------------------------------------------------------------------------------------
# step arrival semantics
# ---------------------------------------------------------------------------------------


def test_step_processes_exactly_one_arrival_event():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.eval()
    state = adapter.init_state(2, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for step in range(3):
            values = torch.randn(2, 3, 8, 8, generator=generator)
            output = adapter.step(
                make_frame(
                    values,
                    camera="a",
                    frame_ids=[step, step],
                    episode_ids=["e0", "e0"],
                    timestamp=[step, step],
                ),
                None,
                state,
                budget(adapter),
            )
            state = output.state
        assert [int(value) for value in state.arrivals["a"].tolist()] == [3, 3]
        assert [int(value) for value in state.unique_arrivals["a"].tolist()] == [3, 3]

        repeat = adapter.step(
            make_frame(
                torch.randn(2, 3, 8, 8, generator=generator),
                camera="a",
                frame_ids=[2, 2],
                episode_ids=["e0", "e0"],
                timestamp=[3, 3],
            ),
            None,
            state,
            budget(adapter),
        )
        assert repeat.diagnostics["repeated_this_step"] == [True, True]
        assert [int(value) for value in repeat.state.arrivals["a"].tolist()] == [4, 4]
        assert [int(value) for value in repeat.state.unique_arrivals["a"].tolist()] == [3, 3]
        assert repeat.diagnostics["window_frames"] == 3

        # A skipped arrival (new_measurement False) advances the clock but adds no evidence.
        skipped = adapter.step(
            make_frame(
                torch.randn(2, 3, 8, 8, generator=generator),
                camera="a",
                frame_ids=[9, 9],
                episode_ids=["e0", "e0"],
                timestamp=[4, 4],
                new_measurement=[False, False],
            ),
            None,
            repeat.state,
            budget(adapter),
        )
        assert [int(value) for value in skipped.state.arrivals["a"].tolist()] == [5, 5]
        assert [int(value) for value in skipped.state.unique_arrivals["a"].tolist()] == [3, 3]

        # readout is not an observation.
        adapter.readout(skipped)
        adapter.readout(skipped)
        assert [int(value) for value in skipped.state.arrivals["a"].tolist()] == [5, 5]
        assert [int(value) for value in skipped.state.unique_arrivals["a"].tolist()] == [3, 3]

        # A new episode resets that member before its first frame is pushed; the other
        # member keeps its buffer because its episode id did not change.
        switched = adapter.step(
            make_frame(
                torch.randn(2, 3, 8, 8, generator=generator),
                camera="a",
                frame_ids=[0, 2],
                episode_ids=["fresh", "e0"],
                timestamp=[5, 5],
            ),
            None,
            skipped.state,
            budget(adapter),
        )
        assert [int(value) for value in switched.state.unique_arrivals["a"].tolist()] == [1, 3]
        assert [int(value) for value in switched.state.arrivals["a"].tolist()] == [1, 6]
        assert [int(value) for value in switched.diagnostics["source_frame_ids"][0].tolist()] == [
            0,
            2,
        ]
        # Member 1 keeps its window; member 0 was reset to a single slot.
        assert [int(value) for value in switched.state.cameras["a"].filled.tolist()] == [1, 3]


# ---------------------------------------------------------------------------------------
# reset / detach semantics
# ---------------------------------------------------------------------------------------


def _leaf_frame(adapter, camera, frame_id, *, seed=0, batch=2):
    generator = torch.Generator()
    values = torch.cat(
        [
            torch.randn(
                1,
                adapter.config.input_channels,
                8,
                8,
                generator=generator.manual_seed(seed * 10 + member),
                requires_grad=True,
            )
            for member in range(batch)
        ],
        dim=0,
    )
    return make_frame(values, camera=camera, frame_ids=[frame_id] * batch)


def test_partial_reset_and_detach_state_keep_the_untouched_member():
    adapter = DbsrFeatureAdapter(tiny_config())
    adapter.train()
    state = adapter.init_state(2, {"camera_ids": ["a", "b"], "shape": (8, 8)}, "cpu")
    output = None
    for step in range(2):
        for camera in ("a", "b"):
            output = adapter.step(
                _leaf_frame(adapter, camera, step, seed=step), None, state, budget(adapter)
            )
            state = output.state
    assert state.cameras["a"].values.requires_grad
    before = {camera: (state.cameras[camera].frame_ids.detach().clone(),) for camera in ("a", "b")}
    before_tokens = adapter.readout(output)

    detached = adapter.detach_state(state)
    assert not detached.cameras["a"].values.requires_grad
    assert detached.tokens["a"].values.grad_fn is None
    assert torch.equal(detached.cameras["a"].values, state.cameras["a"].values)
    assert torch.equal(detached.tokens["a"].values, state.tokens["a"].values)

    partial = adapter.reset(detached, [True, False])
    for camera in ("a", "b"):
        assert int(partial.cameras[camera].filled[0]) == 0
        assert int(partial.cameras[camera].filled[1]) == 2
        assert bool((partial.cameras[camera].frame_ids[:, 0] == -1).all())
        assert torch.equal(partial.cameras[camera].frame_ids[:, 1], before[camera][0][:, 1])
    partial_tokens = adapter.readout(replace(output, state=partial))
    assert not bool(partial_tokens.mask[0].any())
    assert bool(partial_tokens.mask[1].any())
    assert torch.equal(partial_tokens.values[1:2], before_tokens.values[1:2])
    assert partial.episodes["a"] == [None, None]

    cleared = adapter.reset(partial, None)
    assert int(cleared.cameras["b"].filled.max()) == 0
    with pytest.raises(RuntimeError, match="no camera holds a unique measurement"):
        adapter.readout(replace(output, state=cleared))


# ---------------------------------------------------------------------------------------
# channel contract: three-channel stem vs packed-CFA four-channel entry
# ---------------------------------------------------------------------------------------


def test_three_channel_adapter_replaces_stem_and_rejects_four_channels():
    adapter = DbsrFeatureAdapter(tiny_config())
    assert adapter.stem_report() == {
        "input_channels": 3,
        "stem_replaced": True,
        "upstream_default_channels": 4,
        "cfa_entry_channels": 4,
    }
    stem = adapter.backbone.upstream.init_layer[0]
    assert isinstance(stem, nn.Conv2d)
    assert (stem.in_channels, stem.out_channels) == (3, adapter.config.stem_width)
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    with pytest.raises(ValueError, match="expects 3"):
        adapter.step(
            make_frame(torch.randn(1, 4, 8, 8), camera="a", frame_ids=[0]),
            None,
            state,
            budget(adapter),
        )
    with pytest.raises(ValueError, match="channel"):
        adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8), "channels": 4}, "cpu")
    assert P3_SUPPORT_ENABLED is False


def test_cfa_entry_keeps_the_upstream_four_channel_stem_and_old_behaviour():
    burst_config = BurstFeatureConfig(
        stem_width=4, residual_blocks=1, feature_width=8, projection_width=4, offset_width=8
    )
    burst = BurstFeatureEncoder(burst_config)
    adapter = DbsrFeatureAdapter(tiny_config(input_mode="bayer_cfa"))
    assert adapter.stem_report()["stem_replaced"] is False
    assert adapter.backbone.upstream.init_layer[0].in_channels == 4
    # The CFA path must be the untouched upstream branch: identical weights, identical output.
    adapter.backbone.load_state_dict(burst.backbone.state_dict())
    values = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    adapter.eval()
    burst.eval()
    with torch.no_grad():
        assert torch.equal(adapter.backbone(values), burst.backbone(values))
        encoded, centers = burst.encode_frames(
            values[None], torch.ones(1, 2, 4, 8, 8, dtype=torch.bool)
        )
    assert tuple(encoded.shape) == (1, 2, 8, 8, 8)
    assert bool(centers.all())

    canonical = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    state = adapter.init_state(1, {"camera_ids": ["a"], "shape": (8, 8)}, "cpu")
    frame = make_frame(values[:1], camera="a", frame_ids=[0])
    frame.sample_xy = canonical
    with torch.no_grad():
        output = adapter.step(frame, None, state, budget(adapter))
        tokens = adapter.readout(output)
    assert tuple(tokens.values.shape) == (1, 4, 8)

    wrong_phase = canonical.flip(0)
    bad = make_frame(values[:1], camera="a", frame_ids=[1])
    bad.sample_xy = wrong_phase
    with pytest.raises(ValueError, match="canonical RGGB"):
        adapter.step(bad, None, output.state, budget(adapter))
    missing = make_frame(values[:1], camera="a", frame_ids=[1])
    with pytest.raises(ValueError, match="canonical RGGB"):
        adapter.step(missing, None, output.state, budget(adapter))

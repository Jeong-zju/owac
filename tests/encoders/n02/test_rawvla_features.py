"""Contract tests for the π0.5 ↔ OWAC feature bridge (``rawvla_features``).

**These are contract tests, not a π0.5 acceptance.** They run against the stub
backend defined in :mod:`owac.backends.rawvla_features` (``_StubVLM``,
``_StubActionHead``, ``_StubPolicy``). The stub mirrors the pinned module nesting,
state-dict naming, mask/position conventions and flow-matching math, with trivial
attention math; it is a structurally faithful miniature, not the real backend, and
its numbers carry no meaning. They verify what can be verified without the
external runtime: prefix layout, masks, position numbering, the single-encoder-run
guarantee under ten denoising steps, real gradient flow into the encoder while
frozen backend parameters get none, the loading audit for replaced vision weights,
and partial-episode reset.

Tests that genuinely need the pinned StarVLA backend are skipped, and the skip
reason names the missing asset.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from owac.backends.rawvla_features import (
    ACTION_PROTOCOLS,
    LIBERO,
    ROBOTWIN,
    VISION_KEY_PREFIXES,
    ActionChunkCache,
    EncoderReadoutTokenSource,
    RawVLAFeaturePolicy,
    StaticTokenSource,
    TokenSource,
    build_stub_bridge,
    load_audited_replaced_vision,
    protocol_for,
)
from owac.core.encoder_contract import MeasurementFrame, VisualTokens

_STARVLA_AVAILABLE = importlib.util.find_spec("starVLA") is not None
_RAWVLA_ROOT = Path(os.environ.get("RAWVLA_ROOT", "/workspace/owac/rawvla"))
_CHECKPOINT = _RAWVLA_ROOT / "pi05_libero"
_TOKENIZER = _RAWVLA_ROOT / "paligemma_tokenizer.model"

requires_real_backend = pytest.mark.skipif(
    not (_STARVLA_AVAILABLE and _CHECKPOINT.exists()),
    reason=(
        "requires the external π0.5 runtime and the author checkpoint; missing assets here: "
        f"python package 'starVLA' (importable={_STARVLA_AVAILABLE}), "
        f"checkpoint {_CHECKPOINT} (exists={_CHECKPOINT.exists()}), "
        f"tokenizer {_TOKENIZER} (exists={_TOKENIZER.exists()})"
    ),
)


def _language_inputs(batch: int = 2, length: int = 5) -> tuple:
    tokens = torch.arange(length)[None].expand(batch, -1).contiguous()
    return tokens, torch.ones((batch, length), dtype=torch.bool)


# --------------------------------------------------------------------------- #
# Prefix layout, masks and position numbering
# --------------------------------------------------------------------------- #
def test_prefix_layout_masks_and_position_ids() -> None:
    """The prefix reproduces the pinned conventions and masks unsupported tokens."""
    cameras = ("base_0_rgb", "left_wrist_0_rgb")
    batch, per_camera, width = 2, 4, 32
    built = build_stub_bridge(
        width=width, tokens_per_camera=per_camera, cameras=cameras, batch_size=batch
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)

    # camera_ids arrive in the opposite order to image_keys on purpose.
    values = torch.zeros(batch, 2 * per_camera, width)
    values[:, :per_camera] = 1.0  # left_wrist group
    values[:, per_camera:] = -2.0  # base group
    mask = torch.ones((batch, 2 * per_camera), dtype=torch.bool)
    positions = torch.zeros((batch, 2 * per_camera, 2))
    positions[:, :per_camera, 0] = torch.arange(100, 100 + per_camera)  # left_wrist
    positions[:, per_camera:, 0] = torch.arange(per_camera)  # base
    # one unsupported position inside the base group, and one bad value
    positions[0, per_camera + 3, 0] = float("nan")
    mask[1, 1] = False  # a masked token inside the left_wrist group
    values[1, 0, :] = float("nan")  # non-finite measurement

    tokens = VisualTokens(
        values=values, mask=mask, camera_ids=tuple(reversed(cameras)), positions=positions
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)

    assert tuple(prefix.embeds.shape) == (batch, 2 * per_camera + 5, width)
    # image_keys order wins: the base group is first, identified by its positions.
    assert prefix.camera_ids == cameras
    assert torch.equal(prefix.positions[:, 0, 0], torch.zeros(batch))
    assert torch.equal(prefix.positions[:, per_camera, 0], torch.full((batch,), 100.0))
    assert torch.equal(prefix.token_camera, torch.tensor([0] * per_camera + [1] * per_camera))

    expected_mask = torch.ones((batch, 2 * per_camera), dtype=torch.bool)
    expected_mask[0, 3] = False  # base group, unsupported position
    expected_mask[1, per_camera + 0] = False  # left_wrist group, non-finite value
    expected_mask[1, per_camera + 1] = False  # left_wrist group, supplied token mask
    assert torch.equal(prefix.pad_masks[:, : 2 * per_camera], expected_mask)
    assert not prefix.att_masks.any(), "every prefix token must use att_mask 0"
    assert prefix.pad_masks[:, 2 * per_camera :].all(), "language padding is fully supported"

    # masked tokens are zeroed before projection (non-finite value and masked token)
    assert torch.equal(prefix.embeds[1, per_camera + 0], torch.zeros(width))
    assert torch.equal(prefix.embeds[1, per_camera + 1], torch.zeros(width))
    # ... while the unmasked left_wrist tokens keep their value
    assert torch.equal(prefix.embeds[1, per_camera + 2], torch.full((width,), 1.0))
    # image_keys ordering put the -2.0 base values first
    assert torch.allclose(prefix.embeds[:, 0], torch.full((batch, width), -2.0))

    # pinned numbering: cumsum(pad_masks) - 1, so masked tokens consume no position
    position_ids = prefix.position_ids()
    assert torch.equal(position_ids, torch.cumsum(prefix.pad_masks, dim=1) - 1)
    # the first language token's id equals the number of supported visual tokens,
    # so a masked token consumes no position id
    assert int(position_ids[0, 2 * per_camera]) == int(expected_mask[0].sum())
    assert int(position_ids[1, 2 * per_camera]) == int(expected_mask[1].sum())
    assert int(prefix.pad_masks[0, : 2 * per_camera].sum()) == 7
    assert int(prefix.pad_masks[1, : 2 * per_camera].sum()) == 6

    # language embeddings keep the sqrt(hidden) scaling
    lang_emb = bridge.base.vlm.embed_language_tokens(lang_tokens) * (width**0.5)
    assert torch.allclose(prefix.embeds[:, 2 * per_camera :], lang_emb, atol=1e-5)
    assert prefix.summary()["masked_tokens"] == int((~prefix.pad_masks).sum())


def test_feature_prefix_matches_pinned_embed_prefix() -> None:
    """The feature prefix equals a literal transcription of PI0.embed_prefix.

    ``PI0.embed_prefix`` (pinned RawVLA commit 57d508cb) loops over cameras calling
    ``vlm.embed_image``, expands ``image_mask`` to the token count, appends zero
    att_masks for the visual block, scales the language embedding by sqrt(hidden)
    and appends zero att_masks for the language block. The transcription below is
    that code with ``embed_image`` replaced by the OWAC visual tokens, so equality
    is a direct check that the bridge kept the original conventions.
    """
    import math

    cameras = ("base_0_rgb", "left_wrist_0_rgb")
    batch, per_camera, width = 2, 4, 32
    built = build_stub_bridge(
        width=width, tokens_per_camera=per_camera, cameras=cameras, batch_size=batch
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    # The pinned code expresses one scalar image_mask per camera, so the literal
    # comparison uses whole-camera masks; per-token masks are covered separately.
    mask = torch.ones((batch, 2 * per_camera), dtype=torch.bool)
    mask[1, per_camera:] = False  # left_wrist_0_rgb inactive for sample 1
    values = torch.randn(batch, 2 * per_camera, width)
    tokens = VisualTokens(
        values=values.clone(),
        mask=mask,
        camera_ids=cameras,
        positions=torch.zeros((batch, 2 * per_camera, 2)),
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)

    def pinned_embed_prefix(per_camera_embs, img_masks):
        embs, pad_masks, att_masks = [], [], []
        for image_emb, image_mask in zip(per_camera_embs, img_masks, strict=True):
            bsize, num_img_embs = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
        lang_emb = bridge.base.vlm.embed_language_tokens(lang_tokens)
        embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
        return embs, pad_masks, att_masks.expand(pad_masks.shape[0], -1)

    per_camera_embs = [
        values[:, index * per_camera : (index + 1) * per_camera]
        * mask[:, index * per_camera : (index + 1) * per_camera].unsqueeze(-1)
        for index in range(len(cameras))
    ]
    img_masks = [mask[:, index * per_camera] for index in range(len(cameras))]
    expected_embs, expected_pad, expected_att = pinned_embed_prefix(per_camera_embs, img_masks)
    assert torch.equal(prefix.embeds, expected_embs)
    assert torch.equal(prefix.pad_masks, expected_pad)
    assert torch.equal(prefix.att_masks, expected_att)


def test_masked_visual_tokens_cannot_be_attended() -> None:
    """Masked tokens are excluded from the additive attention mask."""
    cameras = ("base_0_rgb", "left_wrist_0_rgb")
    batch, per_camera, width = 2, 4, 32
    built = build_stub_bridge(
        width=width, tokens_per_camera=per_camera, cameras=cameras, batch_size=batch
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    mask = torch.ones((batch, 2 * per_camera), dtype=torch.bool)
    mask[0, per_camera + 1] = False  # invalid camera position / unsupported token
    positions = torch.zeros((batch, 2 * per_camera, 2))
    tokens = VisualTokens(
        values=torch.randn(batch, 2 * per_camera, width),
        mask=mask,
        camera_ids=cameras,
        positions=positions,
    )
    out = bridge.forward_from_visual_tokens(
        tokens,
        lang_tokens,
        lang_masks,
        torch.zeros((batch, 32)),
        bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    attention_mask = out["attention_mask"]
    masked_column = per_camera + 1
    assert torch.all(attention_mask[0, 0, :, masked_column] < -1e30)
    assert torch.any(attention_mask[0, 0, :, masked_column + 1] == 0.0)


def test_missing_camera_contributes_masked_zero_tokens() -> None:
    """A camera with no tokens keeps the layout but is never attended."""
    cameras = ("base_0_rgb", "left_wrist_0_rgb")
    batch, per_camera = 2, 4
    built = build_stub_bridge(
        width=32, tokens_per_camera=per_camera, cameras=cameras, batch_size=batch
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    tokens = VisualTokens(
        values=torch.randn(batch, per_camera, 32),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("base_0_rgb",),
        positions=torch.zeros((batch, per_camera, 2)),
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
    assert tuple(prefix.embeds.shape) == (batch, 2 * per_camera + 5, 32)
    assert prefix.missing_cameras == ("left_wrist_0_rgb",)
    assert not prefix.pad_masks[:, per_camera : 2 * per_camera].any()
    assert torch.equal(
        prefix.embeds[:, per_camera : 2 * per_camera], torch.zeros(batch, per_camera, 32)
    )
    assert any("masked zero tokens" in note for note in prefix.notes)


def test_unknown_camera_and_budget_mismatch_are_errors() -> None:
    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    bad_camera = VisualTokens(
        values=torch.randn(batch, per_camera, 32),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("not_a_camera",),
        positions=torch.zeros((batch, per_camera, 2)),
    )
    with pytest.raises(ValueError, match="outside image_keys"):
        bridge.embed_prefix_from_visual_tokens(bad_camera, lang_tokens, lang_masks)
    wrong_budget = VisualTokens(
        values=torch.randn(batch, per_camera + 1, 32),
        mask=torch.ones((batch, per_camera + 1), dtype=torch.bool),
        camera_ids=("stub",),
        positions=torch.zeros((batch, per_camera + 1, 2)),
    )
    with pytest.raises(ValueError, match="tokens per camera"):
        bridge.embed_prefix_from_visual_tokens(wrong_budget, lang_tokens, lang_masks)


def test_row_major_ordering_option() -> None:
    """positions can re-order tokens inside a camera group when asked."""
    batch, per_camera = 1, 4
    built = build_stub_bridge(
        width=32, tokens_per_camera=per_camera, batch_size=batch, order_within_camera="row_major"
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    positions = torch.zeros((batch, per_camera, 2))
    positions[0, :, 0] = torch.tensor([3.0, 0.0, 2.0, 1.0])
    values = torch.arange(per_camera, dtype=torch.float32)[None, :, None].expand(batch, -1, 32)
    tokens = VisualTokens(
        values=values.clone(),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("stub",),
        positions=positions,
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
    assert torch.equal(prefix.positions[0, :, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prefix.embeds[0, 0], torch.full((32,), 1.0))


# --------------------------------------------------------------------------- #
# One observation, one encoder run
# --------------------------------------------------------------------------- #
def test_encoder_runs_exactly_once_under_ten_denoise_steps() -> None:
    """Ten flow iterations re-read one cached prefix; the encoder runs once."""
    batch, per_camera = 2, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, encoder, source, policy = (
        built["bridge"],
        built["encoder"],
        built["source"],
        built["policy"],
    )
    lang_tokens, lang_masks = _language_inputs(batch)

    prediction = bridge.predict_action(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        noise=torch.zeros((batch, 10, 32)),
        num_steps=10,
    )

    stats = prediction["run_stats"]
    assert stats["prefix_builds"] == 1
    assert stats["token_source_calls"] == 1
    assert stats["encoder_calls"] == 1
    assert stats["denoise_steps"] == 10
    assert source.calls == 1
    assert encoder.step_calls == 1
    assert policy.action_head.denoise_calls == 10
    assert policy.vlm.language_model.prefix_calls == 1
    assert len(set(policy.action_head.cache_tokens)) == 1, "all steps read one KV cache"
    assert prediction["normalized_actions"].shape == (batch, 10, 32)
    assert prediction["control_actions"].shape == (batch, 10, 7)


def test_training_forward_builds_the_prefix_once() -> None:
    batch, per_camera = 2, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, encoder = built["bridge"], built["encoder"]
    lang_tokens, lang_masks = _language_inputs(batch)
    out = bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    assert out["run_stats"]["prefix_builds"] == 1
    assert out["run_stats"]["token_source_calls"] == 1
    assert encoder.step_calls == 1
    assert out["action_loss"].requires_grad, "the training forward must keep the graph"


def test_a_token_source_that_runs_twice_is_rejected() -> None:
    """The single-run guard must actually fire, not just be documented."""

    class _DoubleSource(TokenSource):
        def __init__(self, tokens: VisualTokens) -> None:
            super().__init__()
            self.tokens = tokens

        def __call__(self, observation=None):
            self.calls += 2
            return self.tokens

    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    tokens = VisualTokens(
        values=torch.randn(batch, per_camera, 32),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("stub",),
        positions=torch.zeros((batch, per_camera, 2)),
    )
    bridge = RawVLAFeaturePolicy(
        built["policy"],
        _DoubleSource(tokens),
        protocol=LIBERO,
        token_dim=32,
        tokens_per_camera=per_camera,
        image_keys=("stub",),
    )
    lang_tokens, lang_masks = _language_inputs(batch)
    with pytest.raises(RuntimeError, match="exactly once"):
        bridge.forward(
            observation=None,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=torch.zeros((batch, 32)),
            actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        )


def test_feature_path_never_calls_the_image_tower() -> None:
    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, policy = built["bridge"], built["policy"]
    lang_tokens, lang_masks = _language_inputs(batch)
    bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    assert policy.vlm.image_calls == 0
    with pytest.raises(AssertionError, match="embed_image"):
        policy.vlm.embed_image(torch.zeros(1, 3, 8, 8))


# --------------------------------------------------------------------------- #
# Gradient flow
# --------------------------------------------------------------------------- #
def test_action_loss_gradient_reaches_encoder_and_not_frozen_backend() -> None:
    """The required real gradient-flow verification."""
    batch, per_camera, width = 2, 4, 32
    built = build_stub_bridge(width=width, tokens_per_camera=per_camera, batch_size=batch)
    bridge, encoder = built["bridge"], built["encoder"]
    lang_tokens, lang_masks = _language_inputs(batch)

    assert not bridge.base.vlm.embed_tokens.weight.requires_grad
    assert not bridge.base.action_head.action_in_proj.weight.requires_grad
    assert encoder.stem.weight.requires_grad

    out = bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    out["action_loss"].backward()

    grad = encoder.stem.weight.grad
    assert grad is not None, "no gradient reached the OWAC encoder"
    assert float(grad.abs().sum()) > 0.0, "the encoder gradient is all zeros"
    frozen_with_grad = [
        name
        for name, param in bridge.base.named_parameters()
        if param.grad is not None and float(param.grad.abs().sum()) > 0.0
    ]
    assert frozen_with_grad == [], (
        f"frozen backend parameters received gradient: {frozen_with_grad}"
    )
    assert torch.isfinite(grad).all()


def test_projection_is_trainable_disclosed_and_counted() -> None:
    """A width-mismatched token stream gets a disclosed, counted, trainable projection."""
    batch, per_camera, token_dim, width = 2, 4, 16, 32
    built = build_stub_bridge(
        width=width, token_dim=token_dim, tokens_per_camera=per_camera, batch_size=batch
    )
    bridge, encoder = built["bridge"], built["encoder"]
    lang_tokens, lang_masks = _language_inputs(batch)
    report = bridge.parameter_report()
    assert bridge.projection is not None
    assert report["bridge_added"]["tensors"] == 1
    assert report["bridge_added"]["parameters"] == width * token_dim
    assert "projection" in report["bridge_added"]["names"][0]
    records = {record["name"]: record for record in bridge.trainable_records()}
    assert "bridge.projection.weight" in records
    assert records["bridge.projection.weight"]["parameters"] == width * token_dim

    out = bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    out["action_loss"].backward()
    assert bridge.projection.weight.grad is not None
    assert float(bridge.projection.weight.grad.abs().sum()) > 0.0
    assert float(encoder.stem.weight.grad.abs().sum()) > 0.0


def test_no_projection_when_widths_match() -> None:
    built = build_stub_bridge(width=32, token_dim=32, tokens_per_camera=4, batch_size=1)
    report = built["bridge"].parameter_report()
    assert built["bridge"].projection is None
    assert report["bridge_added"]["parameters"] == 0
    assert "no projection needed" in report["bridge_added"]["disclosure"]
    assert report["token_source_trainable"]["parameters"] > 0


def test_bfloat16_casting_rules() -> None:
    """The pinned bf16 rule: prefix and suffix are cast when the stack is bf16."""
    batch, per_camera = 1, 4
    built = build_stub_bridge(
        width=32, tokens_per_camera=per_camera, batch_size=batch, precision="bfloat16"
    )
    bridge = built["bridge"]
    lang_tokens, lang_masks = _language_inputs(batch)
    tokens = VisualTokens(
        values=torch.randn(batch, per_camera, 32),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("stub",),
        positions=torch.zeros((batch, per_camera, 2)),
    )
    prefix = bridge.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
    assert prefix.embeds.dtype == torch.bfloat16
    out = bridge.forward_from_visual_tokens(
        tokens,
        lang_tokens,
        lang_masks,
        torch.zeros((batch, 32)),
        bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    assert out["v_t"].dtype == torch.float32


def test_inference_uses_inference_mode_and_training_does_not() -> None:
    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, policy = built["bridge"], built["policy"]
    lang_tokens, lang_masks = _language_inputs(batch)
    bridge.forward(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        actions=bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    assert all(policy.grad_enabled_during_train), "training must keep autograd enabled"
    bridge.predict_action(
        observation=built["observation"],
        lang_tokens=lang_tokens,
        lang_masks=lang_masks,
        state=torch.zeros((batch, 32)),
        noise=torch.zeros((batch, 10, 32)),
        num_steps=3,
    )
    assert not any(policy.action_head.grad_enabled_during_denoise)
    assert policy.action_head.denoise_calls == 3
    assert policy.action_head.grad_enabled_during_denoise == [False, False, False]


# --------------------------------------------------------------------------- #
# Action protocol and state lifecycle
# --------------------------------------------------------------------------- #
def test_frozen_action_protocols() -> None:
    assert ACTION_PROTOCOLS[LIBERO].action_horizon == 10
    assert ACTION_PROTOCOLS[LIBERO].replan_steps == 5
    assert ACTION_PROTOCOLS[LIBERO].control_dim == 7
    assert ACTION_PROTOCOLS[ROBOTWIN].action_horizon == 50
    assert ACTION_PROTOCOLS[ROBOTWIN].control_dim == 14
    assert ACTION_PROTOCOLS[ROBOTWIN].action_dim == 32
    # denoising steps are a separate parameter, not derived from the horizon
    assert ACTION_PROTOCOLS[LIBERO].num_denoise_steps == 10
    assert protocol_for("libero") is ACTION_PROTOCOLS[LIBERO]
    with pytest.raises(ValueError, match="unknown action protocol"):
        protocol_for("nope")


def test_protocol_mismatch_with_backend_is_rejected() -> None:
    built = build_stub_bridge(width=32, tokens_per_camera=4, protocol=LIBERO, batch_size=1)
    with pytest.raises(ValueError, match="declares horizon"):
        RawVLAFeaturePolicy(
            built["policy"],
            StaticTokenSource(None),
            protocol=ROBOTWIN,
            token_dim=32,
            tokens_per_camera=4,
            image_keys=("stub",),
        )


def test_control_padding_stays_separate_from_action_dim() -> None:
    built = build_stub_bridge(width=32, tokens_per_camera=4, batch_size=2)
    bridge = built["bridge"]
    padded = bridge.prepare_control_targets(torch.ones(2, 10, 7))
    assert padded.shape == (2, 10, 32)
    assert float(padded[:, :, 7:].abs().sum()) == 0.0
    assert float(padded[:, :, :7].sum()) == 2 * 10 * 7
    assert bridge.control_actions(padded).shape == (2, 10, 7)
    short = bridge.prepare_control_targets(torch.ones(2, 4, 7))
    assert short.shape == (2, 10, 32)
    assert float(short[:, 4:, :].abs().sum()) == 0.0
    with pytest.raises(ValueError, match="exceed horizon"):
        bridge.prepare_control_targets(torch.ones(2, 11, 7))


def test_replan_cadence_and_partial_episode_reset() -> None:
    cache = ActionChunkCache(LIBERO)
    cache.store(0, 0, torch.zeros(10, 7))
    assert not cache.needs_replan(0, 0)
    assert not cache.needs_replan(0, 4)
    assert cache.needs_replan(0, 5)
    assert cache.needs_replan(1, 0)

    batch, per_camera = 2, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, encoder, source = built["bridge"], built["encoder"], built["source"]
    bridge.chunk_cache.store(0, 0, torch.zeros(10, 7))
    bridge.chunk_cache.store(1, 0, torch.zeros(10, 7))
    source.states["stub"] = torch.ones(batch)

    result = bridge.reset_episodes([True, False])

    assert result["action_cache_slots"] == [0]
    assert bridge.chunk_cache.episodes == [1]
    assert encoder.reset_masks[-1] == [True, False]
    assert torch.equal(source.states["stub"], torch.tensor([0.0, 1.0]))
    assert result["token_source"]["episodes"] == [True, False]
    # the unfinished episode keeps its chunk
    assert bridge.chunk_cache.action(1, 3).shape == (7,)


# --------------------------------------------------------------------------- #
# Loading audit with the vision tower replaced
# --------------------------------------------------------------------------- #
def test_load_audit_allows_absent_vision_and_audits_language_actions() -> None:
    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    bridge, policy = built["bridge"], built["policy"]
    full = {key: value.clone() for key, value in policy.state_dict().items()}
    vision_keys = sorted(key for key in full if key.startswith(VISION_KEY_PREFIXES))
    assert vision_keys, "the stub must expose pinned-named vision keys"
    without_vision = {
        key: value for key, value in full.items() if not key.startswith(VISION_KEY_PREFIXES)
    }

    audit = load_audited_replaced_vision(bridge, without_vision, release_vision=False)
    assert audit["replaced_vision_missing"] == vision_keys
    assert audit["replaced_vision_present"] == []
    assert audit["vision_constructed_at_load"]
    assert audit["active_missing"] == []
    assert audit["unexpected"] == []
    assert audit["shape_mismatch"] == []

    # a checkpoint that does contain the vision weights is accepted and reported
    with_vision = load_audited_replaced_vision(bridge, full, release_vision=False)
    assert with_vision["replaced_vision_present"] == vision_keys
    assert with_vision["replaced_vision_missing"] == []

    # language/action parameters are still audited item by item
    tampered_shape = dict(without_vision)
    tampered_shape["vlm.embed_tokens.weight"] = torch.zeros(1, 32)
    with pytest.raises(ValueError, match="does not match pinned StarVLA"):
        load_audited_replaced_vision(bridge, tampered_shape, release_vision=False)
    unexpected = dict(without_vision)
    unexpected["vlm.not_a_real_weight"] = torch.zeros(2)
    with pytest.raises(ValueError, match="does not match pinned StarVLA"):
        load_audited_replaced_vision(bridge, unexpected, release_vision=False)


def test_release_replaced_vision_frees_parameters() -> None:
    built = build_stub_bridge(width=32, tokens_per_camera=4, batch_size=1)
    bridge, policy = built["bridge"], built["policy"]
    released = bridge.release_replaced_vision()
    assert released["released"] is True
    assert released["freed_bytes"] > 0
    assert set(released["released_modules"]) == {"vision_tower", "multi_modal_projector"}
    assert policy.vlm.model.model.vision_tower is None
    assert policy.vlm.vision_tower is None
    remaining = [key for key in policy.state_dict() if key.startswith(VISION_KEY_PREFIXES)]
    assert remaining == [], f"vision parameters still registered: {remaining}"


def test_explicit_token_source_contract() -> None:
    """A caller-supplied static source still runs once and is counted."""
    batch, per_camera = 1, 4
    built = build_stub_bridge(width=32, tokens_per_camera=per_camera, batch_size=batch)
    tokens = VisualTokens(
        values=torch.randn(batch, per_camera, 32),
        mask=torch.ones((batch, per_camera), dtype=torch.bool),
        camera_ids=("stub",),
        positions=torch.zeros((batch, per_camera, 2)),
    )
    source = StaticTokenSource(tokens)
    bridge = RawVLAFeaturePolicy(
        built["policy"],
        source,
        protocol=LIBERO,
        token_dim=32,
        tokens_per_camera=per_camera,
        image_keys=("stub",),
    )
    lang_tokens, lang_masks = _language_inputs(batch)
    out = bridge.forward_from_visual_tokens(
        tokens,
        lang_tokens,
        lang_masks,
        torch.zeros((batch, 32)),
        bridge.prepare_control_targets(torch.randn(batch, 10, 7)),
        noise=torch.randn(batch, 10, 32),
        time=torch.full((batch,), 0.5),
    )
    assert out["run_stats"]["prefix_builds"] == 1
    assert source.calls == 0, "the low-level entry point must not call the token source"


def test_encoder_readout_source_runs_one_step_per_camera() -> None:
    cameras = ("base_0_rgb", "left_wrist_0_rgb")

    class _CountingEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def step(self, frame, context, state, budget):
            from owac.core.encoder_contract import CostRecord, EncoderOutput, Validity

            self.calls += 1
            values = torch.as_tensor(frame.values, dtype=torch.float32)
            return EncoderOutput(
                representation=values.flatten(2).transpose(1, 2),
                state=state,
                validity=Validity(),
                diagnostics={},
                cost=CostRecord(),
            )

        def reset(self, state, batch_mask):
            return state

        def detach_state(self, state):
            return state

    class _Readout:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, output, token_budget):
            from owac.core.encoder_contract import VisualTokens

            self.calls += 1
            values = output.representation
            return VisualTokens(
                values=values,
                mask=torch.ones(values.shape[:2], dtype=torch.bool),
                camera_ids=(),
                positions=torch.zeros((values.shape[0], values.shape[1], 2)),
            )

    encoder, readout = _CountingEncoder(), _Readout()
    source = EncoderReadoutTokenSource(encoder, readout, token_budget=4)
    frames = {
        camera: MeasurementFrame(values=torch.randn(2, 1, 2, 2), camera_id=camera)
        for camera in cameras
    }
    tokens = source({"frames": frames})
    assert encoder.calls == 2 and readout.calls == 2 and source.calls == 1
    assert tokens.camera_ids == cameras
    assert tokens.values.shape == (2, 8, 1)
    assert tokens.positions.shape == (2, 8, 2)


# --------------------------------------------------------------------------- #
# Tests that need the real backend (skipped here)
# --------------------------------------------------------------------------- #
@requires_real_backend
def test_real_pi05_single_batch_forward_backward() -> None:
    """Real π0.5 forward/backward over an OWAC prefix (needs the external runtime)."""
    from owac.backends.rawvla_features import build_feature_policy

    tokens = VisualTokens(
        values=torch.randn(1, 3 * 256, 2048),
        mask=torch.ones((1, 3 * 256), dtype=torch.bool),
        camera_ids=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        positions=torch.zeros((1, 3 * 256, 2)),
    )
    bridge = build_feature_policy(LIBERO, _TOKENIZER, StaticTokenSource(tokens))
    out = bridge.forward_from_visual_tokens(
        tokens,
        torch.zeros((1, 8), dtype=torch.long),
        torch.ones((1, 8), dtype=torch.bool),
        torch.zeros((1, 32)),
        bridge.prepare_control_targets(torch.randn(1, 10, 7)),
    )
    out["action_loss"].backward()
    assert out["action_loss"].requires_grad


@requires_real_backend
def test_real_pi05_predict_runs_encoder_once() -> None:
    """Real π0.5 inference keeps the ten-denoise-steps/one-encoder-run contract."""
    from owac.backends.rawvla_features import build_feature_policy

    calls = {"n": 0}
    tokens = VisualTokens(
        values=torch.randn(1, 3 * 256, 2048),
        mask=torch.ones((1, 3 * 256), dtype=torch.bool),
        camera_ids=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        positions=torch.zeros((1, 3 * 256, 2)),
    )

    class _CountingSource(TokenSource):
        def __call__(self, observation=None):
            self.calls += 1
            calls["n"] += 1
            return tokens

    bridge = build_feature_policy(LIBERO, _TOKENIZER, _CountingSource())
    prediction = bridge.predict_action_from_visual_tokens(
        tokens,
        torch.zeros((1, 8), dtype=torch.long),
        torch.ones((1, 8), dtype=torch.bool),
        torch.zeros((1, 32)),
        num_steps=10,
    )
    assert prediction["run_stats"]["denoise_steps"] == 10
    assert calls["n"] == 0, "the low-level inference entry point must not run the encoder"

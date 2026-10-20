"""Structural native/bridge tracer tests (fixture_contract only)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from owac.evaluation.raw_distillation_trace import (
    BoundaryTrace,
    BridgePolicyTracer,
    NativePolicyTracer,
    ToleranceReceipt,
    compare_boundary_traces,
    run_stage0_traces,
)


class _VLM(torch.nn.Module):
    def embed_image(self, image):
        return image

    def embed_language_tokens(self, tokens):
        return tokens.float().unsqueeze(-1).expand(-1, -1, 6)

    def forward_prefix(self, **kwargs):
        return SimpleNamespace(past_key_values={"fixture": True})


class NativePolicy:
    def __init__(self):
        self.vlm = _VLM()

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        visual = torch.cat([self.vlm.embed_image(image) for image in images], dim=1)
        language = self.vlm.embed_language_tokens(lang_tokens)
        embs = torch.cat([visual, language], dim=1)
        pad = torch.cat(
            [mask[:, None].expand(-1, visual.shape[1] // len(images)) for mask in img_masks]
            + [lang_masks],
            dim=1,
        )
        att = torch.zeros(pad.shape, dtype=torch.bool)
        return embs, pad, att

    def denoise_step(self, state, prefix, past, x_t, timestep):
        return torch.zeros_like(x_t)

    def sample_actions(self, observation, noise=None, num_steps=10):
        images = [observation.images["camera_0"]]
        masks = [observation.image_masks["camera_0"]]
        self.embed_prefix(
            images, masks, observation.tokenized_prompt, observation.tokenized_prompt_mask
        )
        x_t = noise if noise is not None else torch.zeros(1, 10, 32)
        self.vlm.forward_prefix(
            inputs_embeds=torch.zeros(1, 6, 6),
            attention_mask=torch.zeros(1, 1, 6, 6),
            position_ids=torch.arange(6)[None, :],
            past_key_values=None,
            use_cache=True,
        )
        prefix = torch.ones(1, 6, dtype=torch.bool)
        for _ in range(num_steps):
            x_t = x_t + self.denoise_step(None, prefix, None, x_t, torch.tensor(1.0))
        return x_t


class Bridge:
    def __init__(self):
        self.past = None
        self.vlm = _VLM()
        self.base = SimpleNamespace(vlm=self.vlm)

    def mask_2d(self, pad_masks, att_masks):
        return pad_masks[:, None, :] & pad_masks[:, :, None]

    def mask_4d(self, mask):
        return torch.where(mask[:, None], 0.0, -1e9)

    def embed_prefix_from_visual_tokens(self, tokens, lang_tokens, lang_masks):
        language = lang_tokens.float().unsqueeze(-1).expand(-1, -1, 6)
        embs = torch.cat([tokens.values, language], dim=1)
        pad = torch.cat([tokens.mask, lang_masks], dim=1)
        att = torch.zeros(pad.shape, dtype=torch.bool)
        return SimpleNamespace(embeds=embs, pad_masks=pad, att_masks=att)

    def denoise_step(self, *args, **kwargs):
        x_t = args[3] if len(args) > 3 else None
        return torch.zeros_like(x_t)

    def sample_actions_from_visual_tokens(
        self, tokens, lang_tokens, lang_masks, state, noise=None, num_steps=10
    ):
        self.embed_prefix_from_visual_tokens(tokens, lang_tokens, lang_masks)
        x_t = noise if noise is not None else torch.zeros(1, 10, 32)
        self.vlm.forward_prefix(
            inputs_embeds=torch.zeros(1, 6, 6),
            attention_mask=torch.zeros(1, 1, 6, 6),
            position_ids=torch.arange(6)[None, :],
            past_key_values=None,
            use_cache=True,
        )
        prefix = torch.ones(1, 6, dtype=torch.bool)
        for _ in range(num_steps):
            x_t = x_t + self.denoise_step(None, prefix, None, x_t, torch.tensor(1.0))
        return x_t


def _observation():
    return SimpleNamespace(
        images={"camera_0": torch.zeros(1, 4, 6)},
        image_masks={"camera_0": torch.ones(1, dtype=torch.bool)},
        tokenized_prompt=torch.tensor([[1, 2]]),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
        state=torch.zeros(1, 32),
    )


def _stats():
    return SimpleNamespace(action={"q01": [-1.0] * 7, "q99": [1.0] * 7})


def test_native_and_bridge_tracers_capture_required_boundaries_and_compare():
    from owac.core.encoder_contract_n02 import VisualTokens

    native = NativePolicyTracer(NativePolicy(), image_keys=("camera_0",))
    bridge = BridgePolicyTracer(Bridge())
    tokens = VisualTokens(
        values=torch.zeros(1, 4, 6),
        mask=torch.ones(1, 4, dtype=torch.bool),
        camera_ids=("camera_0",),
        positions=torch.zeros(1, 4, 3),
    )
    report = run_stage0_traces(
        native,
        bridge,
        observation=_observation(),
        tokens=tokens,
        lang_tokens=_observation().tokenized_prompt,
        lang_masks=_observation().tokenized_prompt_mask,
        state=_observation().state,
        noise=torch.zeros(1, 10, 32),
        history=None,
        real_stats=_stats(),
        client_postprocess=lambda actions: actions,
        camera_order=("camera_0",),
    )
    assert report.passed
    assert report.tolerance.digest
    assert int(report.native.tensors["denoise.count"]) == 10
    assert report.native.camera_order == ("camera_0",)
    # Missing required execution or action boundary cannot silently pass.
    missing_stats = run_stage0_traces(
        native,
        bridge,
        observation=_observation(),
        tokens=tokens,
        lang_tokens=_observation().tokenized_prompt,
        lang_masks=_observation().tokenized_prompt_mask,
        state=_observation().state,
        noise=torch.zeros(1, 10, 32),
        client_postprocess=lambda actions: actions,
        camera_order=("camera_0",),
    )
    assert missing_stats.passed is False
    assert "actions.real7" in missing_stats.missing_required


def test_tolerance_receipt_is_immutable_floor_plus_multiplier():
    from collections import OrderedDict

    from owac.evaluation.raw_distillation_trace import BoundaryTrace

    first = BoundaryTrace(
        "native",
        OrderedDict(x=torch.ones(1, 2)),
        None,
        False,
        ("camera_0",),
        {},
    )
    second = BoundaryTrace(
        "native",
        OrderedDict(x=torch.ones(1, 2) + 1e-7),
        None,
        False,
        ("camera_0",),
        {},
    )
    receipt = ToleranceReceipt.freeze(first, second, floor=1e-6, multiplier=2.0)
    assert receipt.tolerances["x"] >= 1e-6
    assert receipt.digest


def _compare_trace(*, prefix, pad, img_masks, camera_order=("c0", "c1", "c2"), visual=None):
    from collections import OrderedDict

    tensors = OrderedDict()
    for _index, camera in enumerate(camera_order):
        tensors[f"visual.{camera}"] = visual[camera] if visual is not None else torch.zeros(1, 2, 4)
    tensors["language.tokens"] = torch.zeros(1, 2, dtype=torch.int64)
    tensors["prefix.embeds"] = prefix
    tensors["prefix.pad_masks"] = pad
    tensors["prefix.att_masks"] = torch.zeros_like(pad)
    tensors["input.img_masks"] = torch.as_tensor(img_masks, dtype=torch.bool).reshape(1, -1)
    tensors["forward.attention_mask"] = torch.zeros(1, 1, 1, 1)
    tensors["forward.position_ids"] = torch.arange(4)[None, :]
    tensors["actions.normalized"] = torch.zeros(1, 10, 32)
    tensors["actions.real7"] = torch.zeros(1, 10, 7)
    tensors["actions.executable_first5"] = torch.zeros(1, 5, 7)
    tensors["denoise.count"] = torch.tensor(10)
    return BoundaryTrace(
        "native", tensors, None, False, camera_order, {"inactive_cameras": ("c2",)}
    )


def _tolerance():
    return ToleranceReceipt({}, {}, 1e-6, 2.0, "fixture")


def test_inactive_camera_only_difference_is_explained_without_hiding_raw_delta():
    prefix = torch.arange(1 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4) / 100
    bridge_prefix = prefix.clone()
    bridge_prefix[:, 4:6] = 99.0  # inactive c2 block only
    native = _compare_trace(
        prefix=prefix,
        pad=torch.tensor([[True] * 4 + [False] * 2 + [True] * 2]),
        img_masks=[[True, True, False]],
    )
    bridge = _compare_trace(
        prefix=bridge_prefix,
        pad=torch.tensor([[True] * 4 + [False] * 2 + [True] * 2]),
        img_masks=[[True, True, False]],
    )
    bridge.name = "bridge"
    report = compare_boundary_traces(native, bridge, _tolerance())
    assert report.passed is True
    assert report.raw_prefix_max_delta is not None and report.raw_prefix_max_delta > 0
    assert report.inactive_errors["prefix.embeds"] == report.raw_prefix_max_delta
    assert report.ignored_slot_count == 2
    assert "raw_prefix_full_delta_reported" in report.comparison_policy


def test_active_prefix_and_language_padding_perturbations_fail():
    prefix = torch.zeros(1, 8, 4)
    pad = torch.tensor([[True] * 4 + [False] * 2 + [True] * 2])
    native = _compare_trace(prefix=prefix, pad=pad, img_masks=[[True, True, False]])
    active = prefix.clone()
    active[:, 0] = 1.0
    bridge = _compare_trace(prefix=active, pad=pad, img_masks=[[True, True, False]])
    assert compare_boundary_traces(native, bridge, _tolerance()).passed is False
    lang = prefix.clone()
    lang[:, 6] = 1.0  # language slot, not c2
    bridge = _compare_trace(prefix=lang, pad=pad, img_masks=[[True, True, False]])
    assert compare_boundary_traces(native, bridge, _tolerance()).passed is False


def test_inconsistent_inactive_masks_or_unknown_camera_fail_closed():
    prefix = torch.zeros(1, 8, 4)
    pad = torch.tensor([[True] * 4 + [False] * 2 + [True] * 2])
    native = _compare_trace(prefix=prefix, pad=pad, img_masks=[[True, True, False]])
    bad_active = _compare_trace(prefix=prefix, pad=pad, img_masks=[[True, True, True]])
    assert compare_boundary_traces(native, bad_active, _tolerance()).passed is False
    bad_pad = _compare_trace(
        prefix=prefix, pad=torch.ones_like(pad), img_masks=[[True, True, False]]
    )
    assert compare_boundary_traces(native, bad_pad, _tolerance()).passed is False
    unknown = _compare_trace(
        prefix=prefix, pad=pad, img_masks=[[True, True, False]], camera_order=("c0", "c1")
    )
    unknown.metadata["inactive_cameras"] = ("cX",)
    assert compare_boundary_traces(native, unknown, _tolerance()).passed is False


def test_any_action_difference_still_fails():
    prefix = torch.zeros(1, 8, 4)
    pad = torch.tensor([[True] * 4 + [False] * 2 + [True] * 2])
    native = _compare_trace(prefix=prefix, pad=pad, img_masks=[[True, True, False]])
    bridge = _compare_trace(prefix=prefix, pad=pad, img_masks=[[True, True, False]])
    bridge.tensors["actions.normalized"] = torch.ones(1, 10, 32)
    assert compare_boundary_traces(native, bridge, _tolerance()).passed is False

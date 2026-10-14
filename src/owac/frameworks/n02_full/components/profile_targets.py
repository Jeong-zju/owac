"""Profile targets for the RAW observation encoder (P0 section 6).

Frozen interface (agreed with the dispatcher, recorded in ``components/README.md``):

* :func:`build_profile_targets` returns ``dict[str, torch.nn.Module]``.  Every value is a real
  ``torch.nn.Module`` that can be called directly, and additionally carries a ``profile_meta``
  dict with ``batch_size``, ``sequence_length``, ``input_shape``, ``kind`` and ``notes`` so the
  profile subcommand can find the shapes without unpacking anything.
* :func:`profile_target_specs` returns the same targets in an explicit
  ``{name: {"module": ..., "batch_size": ..., "sequence_length": ..., "input_shape": ...}}``
  form for callers that prefer a description.

Targets are the RAW-adapted ConvNeXt V2 Base encoder (``depths=[3,3,27,3]``,
``dims=[128,256,512,1024]``, 256x256 input) for the 4-plane CFA input and the 3-channel
pseudo-RAW input, plus the FCMAE patch/unpatch geometry op and the FCMAE mask decoder.

``torch`` is imported lazily inside the functions: importing this module must not import torch.
"""

from __future__ import annotations

from typing import Any

BASE_DEPTHS = (3, 3, 27, 3)
BASE_DIMS = (128, 256, 512, 1024)
DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_PATCH_SIZE = 32


class _CallableTarget:
    """Marker base class documented for callers; real targets are plain ``nn.Module``s."""


def _profile_config(config: Any) -> dict[str, Any]:
    import dataclasses

    cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config or {}
    if not isinstance(cfg, dict):
        cfg = {}
    profile = dict(cfg.get("profile", {})) if isinstance(cfg.get("profile"), dict) else {}
    for key in (
        "img_size",
        "batch_size",
        "in_chans",
        "patch_size",
        "decoder_embed_dim",
        "decoder_depth",
        "out_indices",
        "depths",
        "dims",
    ):
        if key in cfg and key not in profile:
            profile[key] = cfg[key]
    return profile


def _as_hw(value) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build_profile_targets(config: Any = None) -> dict[str, Any]:
    """Build the profile targets as ``name -> torch.nn.Module`` (with ``profile_meta``)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(1103)  # frozen component-profile initialization seed
    from .convnextv2 import RawConvNeXtV2
    from .fcmae import MaskAwareDenseConvNeXtV2Encoder, RawFCMAE, patchify, unpatchify

    profile = _profile_config(config)
    img_size = _as_hw(profile.get("img_size", DEFAULT_IMAGE_SIZE))
    batch_size = int(profile.get("batch_size", 1))
    patch_size = int(profile.get("patch_size", DEFAULT_PATCH_SIZE))
    depths = tuple(profile.get("depths", BASE_DEPTHS))
    dims = tuple(profile.get("dims", BASE_DIMS))
    out_indices = tuple(profile.get("out_indices", (1, 2)))
    decoder_embed_dim = int(profile.get("decoder_embed_dim", 512))
    decoder_depth = int(profile.get("decoder_depth", 1))

    targets: dict[str, Any] = {}

    for name, chans, kind in (
        ("convnext_raw4", 4, "cfa_planes"),
        ("convnext_raw3", 3, "pseudo_raw_triplet"),
    ):
        model = RawConvNeXtV2(
            in_chans=chans, depths=depths, dims=dims, num_classes=0, out_indices=out_indices
        )
        model.eval()
        model.profile_meta = {
            "batch_size": batch_size,
            "sequence_length": None,
            "input_shape": (batch_size, chans, img_size[0], img_size[1]),
            "kind": kind,
            "notes": (
                "RAW-adapted ConvNeXt V2 Base observation encoder; profile_meta describes a single "
                "frame batch (no time axis)"
            ),
            "params": int(sum(p.numel() for p in model.parameters())),
        }
        targets[name] = model

    class PatchifyTarget(nn.Module):
        """FCMAE patchify -> unpatchify round trip at the configured patch size."""

        def __init__(self, chans: int, patch: int, hw: tuple[int, int]):
            super().__init__()
            self.chans = chans
            self.patch = patch
            self.hw = hw

        def forward(self, x):
            patches = patchify(x, self.patch)
            return unpatchify(
                patches,
                self.patch,
                self.chans,
                (self.hw[0] // self.patch, self.hw[1] // self.patch),
            )

    patch_target = PatchifyTarget(4, patch_size, img_size)
    patch_target.profile_meta = {
        "batch_size": batch_size,
        "sequence_length": None,
        "input_shape": (batch_size, 4, img_size[0], img_size[1]),
        "kind": "fcmae_patchify_unpatchify",
        "notes": "pure geometry op; also exercised with 3 channels and rectangular inputs",
        "params": 0,
    }
    targets["fcmae_patchify_unpatchify"] = patch_target

    encoder = MaskAwareDenseConvNeXtV2Encoder(
        in_chans=4, depths=depths, dims=dims, patch_size=patch_size
    )
    decoder_target = _MaskDecoderTarget(
        nn_module=nn,
        encoder=encoder,
        dims=dims,
        patch_size=patch_size,
        in_chans=4,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
    )
    feature_hw = (img_size[0] // patch_size, img_size[1] // patch_size)
    decoder_target.profile_meta = {
        "batch_size": batch_size,
        "sequence_length": None,
        "input_shape": (batch_size, dims[-1], feature_hw[0], feature_hw[1]),
        "kind": "fcmae_mask_decoder",
        "notes": (
            "FCMAE decoder half: 1x1 projection, mask-token insertion, decoder Blocks and the "
            "pixel head; input is the encoder feature map at stride patch_size"
        ),
        "params": int(sum(p.numel() for p in decoder_target.parameters())),
    }
    targets["fcmae_mask_decoder"] = decoder_target

    full_fcmae = RawFCMAE(
        img_size=img_size,
        in_chans=4,
        depths=depths,
        dims=dims,
        decoder_depth=decoder_depth,
        decoder_embed_dim=decoder_embed_dim,
        patch_size=patch_size,
    )
    full_fcmae.eval()
    full_fcmae.profile_meta = {
        "batch_size": batch_size,
        "sequence_length": None,
        "input_shape": (batch_size, 4, img_size[0], img_size[1]),
        "kind": "fcmae_full_forward",
        "notes": "full FCMAE forward (masked encoder + decoder + reconstruction loss)",
        "params": int(sum(p.numel() for p in full_fcmae.parameters())),
    }
    targets["fcmae_full"] = full_fcmae

    from .savi import SlotAttention, TransformerPredictor

    class SaviTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.corrector = SlotAttention(num_iterations=1, qkv_size=512, mlp_size=1024).build(
                512, 512
            )
            self.predictor = TransformerPredictor()

        def forward(self, frames):
            state = frames.new_zeros((frames.shape[0], 16, 512))
            outputs = []
            for frame in frames.unbind(1):
                state = self.predictor(self.corrector(state, frame))
                outputs.append(state)
            return __import__("torch").stack(outputs, dim=1)

    savi = SaviTarget()
    sequence_length = int(profile.get("sequence_length", 8))
    savi.profile_meta = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "input_shape": (batch_size, sequence_length, 256, 512),
        "kind": "savi_512_16slots_4predictor_blocks",
        "notes": "256 input tokens per frame; full short-sequence backpropagation",
        "params": sum(p.numel() for p in savi.parameters()),
    }
    targets["savi_512"] = savi

    for target in targets.values():
        target.profile_meta["operation"] = (
            profile.get("operation", "train_step") if target.profile_meta["params"] else "forward"
        )
        target.profile_meta["input_distribution"] = "normal_seed_1103"
    return targets


def _MaskDecoderTarget(
    nn_module,
    encoder,
    dims,
    patch_size: int,
    in_chans: int,
    decoder_embed_dim: int,
    decoder_depth: int,
):
    """Reuse the FCMAE decoder half as a standalone module (no module-level torch import)."""
    import torch

    from .convnextv2 import Block

    class MaskDecoder(nn_module.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn_module.Conv2d(dims[-1], decoder_embed_dim, kernel_size=1)
            self.mask_token = nn_module.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
            self.decoder = nn_module.Sequential(
                *[Block(dim=decoder_embed_dim, drop_path=0.0) for _ in range(decoder_depth)]
            )
            self.pred = nn_module.Conv2d(decoder_embed_dim, patch_size**2 * in_chans, kernel_size=1)
            self.patch_size = patch_size

        def forward(self, features, mask):
            x = self.proj(features)
            n, _c, h, w = x.shape
            mask = mask.reshape(n, h, w).unsqueeze(1).type_as(x)
            token = self.mask_token.repeat(n, 1, h, w)
            x = x * (1.0 - mask) + token * mask
            return self.pred(self.decoder(x))

    return MaskDecoder()


def profile_target_specs(config: Any = None) -> dict[str, dict[str, Any]]:
    """Same targets in an explicit descriptive form (module + batch/time/shape metadata)."""
    targets = build_profile_targets(config)
    return {
        name: {
            "module": module,
            "batch_size": module.profile_meta["batch_size"],
            "sequence_length": module.profile_meta["sequence_length"],
            "input_shape": module.profile_meta["input_shape"],
            "kind": module.profile_meta["kind"],
            "notes": module.profile_meta["notes"],
            "params": module.profile_meta["params"],
        }
        for name, module in targets.items()
    }


__all__ = [
    "BASE_DEPTHS",
    "BASE_DIMS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_PATCH_SIZE",
    "build_profile_targets",
    "profile_target_specs",
]

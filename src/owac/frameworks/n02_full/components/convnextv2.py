"""RAW-adapted ConvNeXt V2 (R01) observation encoder.

Upstream (verbatim copy kept next to this file)::

    components/upstream/convnextv2/models/convnextv2.py
    components/upstream/convnextv2/models/utils.py
    repository facebookresearch/ConvNeXt-V2 @ 2553895753323c6fe0b2bf390683f5ea358a42b9

Port differences against the upstream file (also recorded in
``configs/encoders/n02_full/third_party.lock.json``):

* ``timm`` is not importable in the project root environment, so ``trunc_normal_`` and
  ``DropPath`` are re-implemented here with timm-equivalent semantics instead of being
  imported.  ``LayerNorm``/``GRN``/``Block`` keep the upstream math unchanged.
* ``in_chans`` is honoured end-to-end (upstream already does this for the dense model; the
  hard-coded 3-channel assumption lives in ``models/fcmae.py``, see ``fcmae.py``).
* ``num_classes=0`` builds a head-less encoder; the module then exposes multiscale feature
  maps (stride 4/8/16/32) instead of the pooled classification feature.
* ``forward_stagewise`` returns every intermediate activation so that parity tests can compare
  layer by layer; the upstream ``forward_features``/``forward`` behaviour is preserved.
* No classification head is used by the project; ``head`` is only created when
  ``num_classes > 0`` so that a full upstream state_dict can still be loaded with
  ``strict=True``.

Nothing here is a re-design of the block: with identical weights and ``in_chans=3`` the output
must equal ``upstream.models.convnextv2.Block`` / ``ConvNeXtV2`` bit-for-bit within the FP32
parity tolerance (see ``parity_suite.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

UPSTREAM_REPOSITORY = "facebookresearch/ConvNeXt-V2"
UPSTREAM_COMMIT = "2553895753323c6fe0b2bf390683f5ea358a42b9"

DEFAULT_DEPTHS = (3, 3, 27, 3)
DEFAULT_DIMS = (128, 256, 512, 1024)
# stage index -> spatial stride; stage 0 is produced by the 4x4/stride-4 stem.
STAGE_STRIDES = (4, 8, 16, 32)


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Truncated normal init with the semantics of ``timm.models.layers.trunc_normal_``.

    Values outside ``[mean + a * std, mean + b * std]`` are resampled from a uniform
    distribution restricted to the truncated tails, matching timm's ``_no_grad_trunc_normal_``.
    """
    if std == 0:
        raise ValueError("std must be non-zero")
    with torch.no_grad():
        tensor.normal_(mean, std)
        while True:
            mask = None
            if a is not None:
                mask = tensor < (mean + a * std)
            if b is not None:
                b_mask = tensor > (mean + b * std)
                mask = b_mask if mask is None else (mask | b_mask)
            if mask is None or not mask.any():
                break
            std_ratio = float(b - a) / float(std) if std != 0 else 0.0
            if (a is not None and b is not None) and std_ratio > 0:
                vals = torch.zeros(mask.sum(), device=tensor.device, dtype=tensor.dtype)
                vals.uniform_(mean + a * std, mean + b * std)
                tensor[mask] = vals
            else:
                tensor[mask] = tensor[mask].normal_(mean, std)
    return tensor


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Per-sample stochastic depth, equivalent to ``timm.models.layers.drop_path``."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    keep = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        keep.div_(keep_prob)
    return x * keep


class DropPath(nn.Module):
    """``timm.models.layers.DropPath`` replacement (no timm dependency)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class LayerNorm(nn.Module):
    """Upstream ``models.utils.LayerNorm`` (channels_last / channels_first)."""

    def __init__(self, normalized_shape, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        raise NotImplementedError


class GRN(nn.Module):
    """Upstream ``models.utils.GRN`` (Global Response Normalization).

    The statistics are taken over the full spatial extent, per sample and channel.  That
    spatial coupling is deliberate and is asserted by the parity suite (a single distant pixel
    must change every output pixel).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """Upstream ``models.convnextv2.Block`` (dense, channels-last inner layout)."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + self.drop_path(x)


class RawConvNeXtV2(nn.Module):
    """ConvNeXt V2 backbone adapted for RAW observation input.

    Args:
        in_chans: sensor planes. 3 for the RawVLA pseudo-RAW triplet, 4 for CFA planes.
        depths/dims: stage depth and width; formulation starts from the public Base sizes
            ``[3, 3, 27, 3]`` / ``[128, 256, 512, 1024]``.
        out_indices: stage indices returned by :meth:`forward_features`; default ``(1, 2)``
            keeps stride 8 and stride 16 (contract section 5).
        num_classes: 0 builds a head-less encoder (``head`` is None).
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = DEFAULT_DEPTHS,
        dims: Sequence[int] = DEFAULT_DIMS,
        drop_path_rate: float = 0.0,
        num_classes: int = 0,
        out_indices: Sequence[int] = (1, 2),
        head_init_scale: float = 1.0,
        init_weights: bool = True,
    ):
        super().__init__()
        depths = tuple(int(d) for d in depths)
        dims = tuple(int(d) for d in dims)
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("only the 4-stage ConvNeXt V2 layout is supported")
        self.in_chans = int(in_chans)
        self.depths = depths
        self.dims = dims
        self.out_indices = tuple(int(i) for i in out_indices)
        if any(i < 0 or i > 3 for i in self.out_indices):
            raise ValueError("out_indices must be within 0..3")

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else None

        if init_weights:
            self.apply(self._init_weights)
            if self.head is not None:
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_stagewise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every intermediate activation, keyed by module name, for layer-wise parity."""
        out: dict[str, torch.Tensor] = {"input": x}
        for i in range(4):
            x = self.downsample_layers[i](x)
            out[f"downsample_layers.{i}"] = x
            x = self.stages[i](x)
            out[f"stages.{i}"] = x
        out["pooled"] = self.norm(x.mean([-2, -1]))
        return out

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Selected stage feature maps (default stride 8/16), as a list in ``out_indices`` order."""
        stages: list[torch.Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            stages.append(x)
        return [stages[i] for i in self.out_indices]

    def forward_multiscale(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Named multiscale features: ``stem`` plus ``stage{i}_stride{s}`` for every stage."""
        feats: dict[str, torch.Tensor] = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i == 0:
                feats["stem"] = x
            x = self.stages[i](x)
            feats[f"stage{i}_stride{STAGE_STRIDES[i]}"] = x
        return feats

    def forward(self, x: torch.Tensor):
        """Upstream-compatible forward: pooled feature, and logits when a head exists."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        if self.head is not None:
            return self.head(x)
        return x

    @staticmethod
    def num_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def raw_convnextv2(
    variant: str = "base",
    in_chans: int = 3,
    **kwargs,
) -> RawConvNeXtV2:
    """Convenience constructor; ``variant`` selects one of the public size presets."""
    presets = {
        "atto": ([2, 2, 6, 2], [40, 80, 160, 320]),
        "femto": ([2, 2, 6, 2], [48, 96, 192, 384]),
        "pico": ([2, 2, 6, 2], [64, 128, 256, 512]),
        "nano": ([2, 2, 8, 2], [80, 160, 320, 640]),
        "tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
        "base": (list(DEFAULT_DEPTHS), list(DEFAULT_DIMS)),
        "large": ([3, 3, 27, 3], [192, 384, 768, 1536]),
    }
    if variant not in presets:
        raise KeyError(f"unknown ConvNeXt V2 variant {variant!r}")
    depths, dims = presets[variant]
    return RawConvNeXtV2(in_chans=in_chans, depths=depths, dims=dims, **kwargs)


__all__ = [
    "Block",
    "DEFAULT_DEPTHS",
    "DEFAULT_DIMS",
    "DropPath",
    "GRN",
    "LayerNorm",
    "RawConvNeXtV2",
    "STAGE_STRIDES",
    "drop_path",
    "raw_convnextv2",
    "trunc_normal_",
]

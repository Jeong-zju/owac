"""FCMAE and the mask-aware dense encoder port (R01).

Upstream (verbatim copies kept next to this file)::

    components/upstream/convnextv2/models/fcmae.py
    components/upstream/convnextv2/models/convnextv2_sparse.py
    repository facebookresearch/ConvNeXt-V2 @ 2553895753323c6fe0b2bf390683f5ea358a42b9

Known upstream defects fixed here (audit item 1 of ``01_CURRENT_IMPLEMENTATION.md``):

* ``FCMAE.patchify``/``unpatchify`` hard-code 3 channels (``x.reshape(..., 3, h, p, w, p)``)
  and ``unpatchify`` assumes ``h == w == sqrt(L)``.  This port takes ``in_chans`` and an explicit
  ``patch_grid`` so CFA4 and 240x320 rectangular inputs round-trip exactly.
* ``upsample_mask`` assumes a square patch grid; the port takes the patch grid explicitly.
* ``gen_random_mask`` counts patches from ``shape[2]`` only; the port counts from both axes.

Sparse encoder status: ``models/convnextv2_sparse.py`` imports ``MinkowskiEngine`` (and
``MinkowskiOps.to_sparse``) at module scope, which is not installable in this environment; the
attempt and its raw output are kept in ``outputs/n02-full/p0/parity/logs/``.  The upstream sparse
operator therefore has **no parity receipt** in P0.  What is provided instead, as explicitly
allowed by the P0 contract section 5.2, is

* :class:`MaskAwareDenseConvNeXtV2Encoder` -- a *dense* port of the sparse encoder that keeps the
  same structure, non-expanding stride-one coordinates, strided downsample support and
  valid-only
  normalization statistics.  It is a **naming reference, not the upstream operator**: its compute
  and memory cost are those of a dense network, so it must not be used to claim sparse efficiency.
* :class:`PlainDenseZeroFillEncoder` -- the plain zero-filled dense control, kept under its own
  name and never described as the sparse encoder.

Any statement about the upstream sparse encoder in the P0 report is limited to "not run".
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnextv2 import GRN, Block, LayerNorm, trunc_normal_

UPSTREAM_REPOSITORY = "facebookresearch/ConvNeXt-V2"
UPSTREAM_COMMIT = "2553895753323c6fe0b2bf390683f5ea358a42b9"


def _as_hw(img_size) -> tuple[int, int]:
    if isinstance(img_size, (tuple, list)):
        if len(img_size) != 2:
            raise ValueError("img_size must be an int or a (height, width) pair")
        return int(img_size[0]), int(img_size[1])
    return int(img_size), int(img_size)


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(N, C, H, W) -> (N, H/p * W/p, p*p*C)``; channels are general, H/W need not be equal."""
    p = int(patch_size)
    n, c, h, w = imgs.shape
    if h % p or w % p:
        raise ValueError(f"height/width {h}x{w} not divisible by patch_size {p}")
    gh, gw = h // p, w // p
    x = imgs.reshape(n, c, gh, p, gw, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(n, gh * gw, p * p * c)


def unpatchify(
    patches: torch.Tensor, patch_size: int, in_chans: int, patch_grid=None
) -> torch.Tensor:
    """Inverse of :func:`patchify`; ``patch_grid=(gh, gw)`` disambiguates rectangular grids."""
    p = int(patch_size)
    n, length, width = patches.shape
    if width != p * p * in_chans:
        raise ValueError(f"last dim {width} != patch_size**2 * in_chans {p * p * in_chans}")
    if patch_grid is None:
        side = int(round(length**0.5))
        if side * side != length:
            raise ValueError("non-square patch grid requires an explicit patch_grid=(gh, gw)")
        gh = gw = side
    else:
        gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if gh * gw != length:
        raise ValueError(f"patch_grid {gh}x{gw} does not match {length} patches")
    x = patches.reshape(n, gh, gw, p, p, in_chans)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(n, in_chans, gh * p, gw * p)


def gen_random_mask(
    patch_grid,
    batch_size: int,
    mask_ratio: float,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Upstream mask generation with a rectangular grid: 0 keeps a patch, 1 removes it.

    Exactly ``int(L * (1 - mask_ratio))`` patches are kept per sample, chosen uniformly at
    random over the flattened patch index.
    """
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    length = gh * gw
    len_keep = int(length * (1 - mask_ratio))
    noise = torch.randn(batch_size, length, device=device, generator=generator)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones(batch_size, length, device=device)
    mask[:, :len_keep] = 0
    return torch.gather(mask, dim=1, index=ids_restore)


def upsample_patch_mask(mask: torch.Tensor, scale: int, patch_grid) -> torch.Tensor:
    """Patch-grid mask ``(N, L)`` -> pixel mask ``(N, gh*scale, gw*scale)`` (0 keep / 1 remove)."""
    gh, gw = int(patch_grid[0]), int(patch_grid[1])
    if mask.shape[1] != gh * gw:
        raise ValueError(f"mask has {mask.shape[1]} entries, expected {gh * gw}")
    m = mask.reshape(-1, gh, gw)
    m = m.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
    return m


def dilate_support(valid: torch.Tensor, kernel: int, stride: int = 1) -> torch.Tensor:
    """Generic coordinate dilation; NOT the support rule of stride-one Minkowski convolution.

    An output position is valid iff its convolution window contains at least one valid input
    position.  ``valid`` is ``(N, 1, H, W)`` (or any ``(N, C, H, W)`` broadcastable map).
    """
    if kernel < 1:
        raise ValueError("kernel must be >= 1")
    if kernel == 1 and stride == 1:
        return valid
    if kernel == stride:
        # Non-overlapping windows: max-pool is exactly the dilation.
        pooled = F.max_pool2d(valid, kernel_size=kernel, stride=stride)
        return (pooled > 0).to(valid.dtype)
    # Overlapping windows need an exhaustive elementwise max over shifted copies.
    pad = (kernel - 1) // 2
    padded = F.pad(valid, (pad, pad, pad, pad), value=0)
    out = None
    for dy in range(kernel):
        for dx in range(kernel):
            window = padded[
                :, :, dy : dy + valid.shape[-2] : stride, dx : dx + valid.shape[-1] : stride
            ]
            out = window if out is None else torch.maximum(out, window)
    return (out > 0).to(valid.dtype)


class MaskAwareDenseConvNeXtV2Encoder(nn.Module):
    """Dense port of ``convnextv2_sparse.SparseConvNeXtV2`` with explicit support bookkeeping.

    Structure and per-layer math follow the upstream sparse encoder:
    dense stem (Conv4x4/s4 + channels-first LayerNorm), mask re-application at stem resolution,
    then three ``LayerNorm -> Conv2x2/s2`` downsampling steps interleaved with four stages of
    depthwise-7x7 blocks.  The only intentional differences are

    * convolutions run densely on the zero-filled support (identical arithmetic at valid
      positions, different cost -- documented as a naming reference, not the upstream operator);
    * ``LayerNorm``/``GRN`` statistics are computed over valid positions only, matching
      ``MinkowskiLayerNorm``/``MinkowskiGRN``;
    * ``GRN`` reduces the channel norm over the whole batch by default
      (``grn_scope="sparse"``), matching ``MinkowskiGRN``; ``grn_scope="per_sample"`` matches the
      dense ``models.utils.GRN`` used by the dense network.

    ``mask`` follows the upstream convention: 1 removes, 0 keeps.
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 27, 3),
        dims: Sequence[int] = (128, 256, 512, 1024),
        drop_path_rate: float = 0.0,
        grn_scope: str = "sparse",
        patch_size: int = 32,
        init_weights: bool = True,
    ):
        super().__init__()
        depths = tuple(int(d) for d in depths)
        dims = tuple(int(d) for d in dims)
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("only the 4-stage layout is supported")
        if grn_scope not in ("sparse", "per_sample"):
            raise ValueError("grn_scope must be 'sparse' or 'per_sample'")
        if int(patch_size) != 4 * 2 ** (len(depths) - 1):
            raise ValueError(
                f"patch_size {patch_size} does not match the stem stride 4 times {len(depths)} "
                f"downsampling stages (expected {4 * 2 ** (len(depths) - 1)})"
            )
        self.in_chans = int(in_chans)
        self.depths = depths
        self.dims = dims
        self.grn_scope = grn_scope
        self.patch_size = int(patch_size)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList(
                [MaskAwareBlock(dims[i], dp_rates[cur + j], grn_scope) for j in range(depths[i])]
            )
            self.stages.append(blocks)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """Returns ``(features, valid, stages)``: dense map, its support, and per-stage supports."""
        patch_scale = self.patch_size // 4
        gh, gw = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patch_grid = (gh, gw)

        valid = 1.0 - upsample_patch_mask(mask, patch_scale, patch_grid).unsqueeze(1).type_as(x)
        x = self._apply_conv(self.downsample_layers[0][0], x)
        x = self.downsample_layers[0][1](x)
        x = x * valid
        supports = [valid]

        for i in range(4):
            if i > 0:
                x = self._masked_channels_first_layernorm(self.downsample_layers[i][0], x, valid)
                x = self._apply_conv(self.downsample_layers[i][1], x)
                valid = dilate_support(valid, kernel=2, stride=2)
                x = x * valid
                supports.append(valid)
            for block in self.stages[i]:
                x, valid = block(x, valid)
        return x, valid, supports

    @staticmethod
    def _apply_conv(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, conv.weight, conv.bias, stride=conv.stride, padding=0)

    @staticmethod
    def _masked_channels_first_layernorm(ln: LayerNorm, x: torch.Tensor, valid: torch.Tensor):
        # MinkowskiLayerNorm applies nn.LayerNorm to each active feature row:
        # normalize channels per coordinate, never spatial positions per channel.
        return ln(x) * valid


class MaskAwareBlock(nn.Module):
    """``convnextv2_sparse.Block`` with the same parameter names and valid-only statistics."""

    def __init__(self, dim: int, drop_path: float = 0.0, grn_scope: str = "sparse"):
        super().__init__()
        # Dense zero padding gives the same neighbor sum at the existing sparse coordinates.
        # Mask after every block so bias at absent coordinates cannot enter later convolutions.
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()  # upstream sparse blocks apply no drop path at P0 eval
        self._drop_path_rate = drop_path
        self.grn_scope = grn_scope
        self._last_stats: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        residual = x * valid
        x = F.conv2d(residual, self.dwconv.weight, self.dwconv.bias, padding=3, groups=x.shape[1])
        # expand_coordinates=False: depthwise stride-one convolution preserves support.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self._masked_grn(x, valid)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        out = residual + x
        return out * valid, valid

    def _masked_grn(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """GRN with statistics restricted to valid positions (``MinkowskiGRN`` semantics).

        ``torch.linalg.vector_norm`` is used instead of ``sqrt(sum(...))`` on purpose: upstream
        ``MinkowskiGRN``/``GRN`` call ``torch.norm``, whose backward at zero is defined (0), while
        the backward of ``sqrt`` at zero is infinite.  Using ``sqrt`` here produced NaN gradients
        for all-zero inputs, which the upstream operators do not.
        """
        v = valid.permute(0, 2, 3, 1)  # (N, H, W, 1)
        if self.grn_scope == "sparse":
            count = v.sum().clamp_min(1.0)
            Gx = torch.linalg.vector_norm(x * v, dim=(0, 1, 2), keepdim=True)
        else:
            count = v.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            Gx = torch.linalg.vector_norm(x * v, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        self._last_stats = {
            "Gx": Gx.detach(),
            "Nx": Nx.detach(),
            "count": count.detach(),
            "pre_grn_x": x.detach(),
            "valid": v.detach(),
        }
        return self.grn.gamma * (x * Nx) + self.grn.beta + x

    def masked_grn_params(self):
        return {"gamma": self.grn.gamma, "beta": self.grn.beta}


class PlainDenseZeroFillEncoder(nn.Module):
    """Plain zero-filled dense control (named control, **not** the upstream sparse encoder).

    Masked patches are zeroed and the ordinary dense ConvNeXt V2 stages are applied, with GRN
    statistics taken over every position (including the zero-filled ones).  Used only to show
    that mask-aware valid-only statistics change the result.
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 27, 3),
        dims: Sequence[int] = (128, 256, 512, 1024),
        init_weights: bool = True,
    ):
        super().__init__()
        depths = tuple(int(d) for d in depths)
        dims = tuple(int(d) for d in dims)
        self.depths = depths
        self.dims = dims
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], 0.0) for _ in range(depths[i])]))
        if init_weights:
            self.apply(_dense_init)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        patch_scale = 8
        gh, gw = x.shape[-2] // 32, x.shape[-1] // 32
        pixel_mask = upsample_patch_mask(mask, patch_scale, (gh, gw)).unsqueeze(1).type_as(x)
        x = self.downsample_layers[0](x) * (1.0 - pixel_mask)
        for i in range(4):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


def _dense_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class RawFCMAE(nn.Module):
    """FCMAE adapted for general channel counts and rectangular inputs.

    ``img_size`` may be an int or ``(height, width)``; ``in_chans`` may be 3 (RawVLA pseudo-RAW
    triplet) or 4 (CFA planes).  ``encoder`` defaults to the mask-aware dense port; pass any module
    with the ``(x, mask) -> (features, valid, supports)`` signature.
    """

    def __init__(
        self,
        img_size=(256, 256),
        in_chans: int = 3,
        depths: Sequence[int] = (3, 3, 27, 3),
        dims: Sequence[int] = (128, 256, 512, 1024),
        decoder_depth: int = 1,
        decoder_embed_dim: int = 512,
        patch_size: int = 32,
        mask_ratio: float = 0.6,
        norm_pix_loss: bool = False,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.img_size = _as_hw(img_size)
        self.in_chans = int(in_chans)
        self.depths = tuple(int(d) for d in depths)
        self.dims = tuple(int(d) for d in dims)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.num_patches = (self.img_size[0] // self.patch_size) * (
            self.img_size[1] // self.patch_size
        )
        self.decoder_embed_dim = int(decoder_embed_dim)
        self.decoder_depth = int(decoder_depth)
        self.norm_pix_loss = bool(norm_pix_loss)

        if encoder is None:
            encoder = MaskAwareDenseConvNeXtV2Encoder(
                self.in_chans, depths=self.depths, dims=self.dims
            )
        self.encoder = encoder
        self.proj = nn.Conv2d(self.dims[-1], self.decoder_embed_dim, kernel_size=1)
        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_embed_dim, 1, 1))
        self.decoder = nn.Sequential(
            *[Block(dim=self.decoder_embed_dim, drop_path=0.0) for _ in range(self.decoder_depth)]
        )
        self.pred = nn.Conv2d(
            self.decoder_embed_dim, self.patch_size**2 * self.in_chans, kernel_size=1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        return patchify(imgs, self.patch_size)

    def unpatchify(self, x: torch.Tensor, patch_grid=None) -> torch.Tensor:
        if patch_grid is None:
            gh = self.img_size[0] // self.patch_size
            gw = self.img_size[1] // self.patch_size
            patch_grid = (gh, gw)
        return unpatchify(x, self.patch_size, self.in_chans, patch_grid)

    def patch_grid_for(self, height: int, width: int) -> tuple[int, int]:
        return height // self.patch_size, width // self.patch_size

    def gen_random_mask(self, x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        gh, gw = self.patch_grid_for(x.shape[2], x.shape[3])
        return gen_random_mask((gh, gw), x.shape[0], mask_ratio, device=x.device)

    def upsample_mask(self, mask: torch.Tensor, scale: int) -> torch.Tensor:
        side = int(round(mask.shape[1] ** 0.5))
        patch_grid = (side, side)
        if side * side != mask.shape[1]:
            raise ValueError("non-square patch grid requires an explicit grid")
        return upsample_patch_mask(mask, scale, patch_grid)

    def forward_encoder(self, imgs: torch.Tensor, mask_ratio: float):
        mask = self.gen_random_mask(imgs, mask_ratio)
        x, valid, supports = self.encoder(imgs, mask)
        return x, mask, valid, supports

    def forward_decoder(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        n, c, h, w = x.shape
        mask = mask.reshape(n, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(n, 1, h, w)
        x = x * (1.0 - mask) + mask_token * mask
        x = self.decoder(x)
        return self.pred(x)

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """Reconstruction loss on removed patches only; masked pixels never enter the target."""
        if pred.ndim == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1).permute(0, 2, 1)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, imgs: torch.Tensor, mask_ratio: float | None = None):
        ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        x, mask, valid, supports = self.forward_encoder(imgs, ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


__all__ = [
    "MaskAwareBlock",
    "MaskAwareDenseConvNeXtV2Encoder",
    "PlainDenseZeroFillEncoder",
    "RawFCMAE",
    "dilate_support",
    "gen_random_mask",
    "patchify",
    "unpatchify",
    "upsample_patch_mask",
]

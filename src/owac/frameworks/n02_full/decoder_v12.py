"""N02 v1.2 conditional RAW measurement decoder.

Limited PyTorch port of the official SAVi spatial broadcast decoder, adapted to predict a
**measurement** (mean + variance + condition masks) instead of an RGB frame.

Upstream source (Apache-2.0, vendored byte-for-byte under ``third_party/pinned``)::

    repository google-research/slot-attention-video @ ba8f15ee19472c6f9425c9647daf87910f17b605
    savi/modules/decoders.py   SpatialBroadcastDecoder
    savi/modules/convolution.py CNN            (decoder backbone)
    savi/modules/misc.py        PositionEmbedding, Readout
    savi/modules/initializers.py / configs/movi/savi_conditional_medium.py
                                (MoVi decoder configuration: CNN features [64,64,64,64],
                                 kernel 5x5 / stride 2 transposed convs, lecun-normal init)

Port differences (all deliberate and listed here rather than hidden in the code):

* ``functools.partial(utils.time_distributed, in_axes=1)`` (vmap over the time axis) is
  replaced by an explicit **camera** axis: this decoder is applied once per camera with the
  broadcast grid at that camera's own stride-4 RAW resolution, so cameras keep their own
  resolution and are never folded into the batch axis.
* The slot axis keeps the official semantics (``alpha = softmax(alpha_logits, axis=slots)``,
  per-slot target channels combined by ``sum_slot(channels * alpha)``).  A *candidate* in
  N02 is one self-consistent world interpretation with ``n_slots`` slots, so the decoder is
  called with ``(B*K, N, slot_dim)`` slots and candidate ``k`` never mixes with candidate
  ``k'``.
* Outputs are RAW measurement targets (mean, log-variance, condition logits) in the camera's
  sensor units, not RGB.  There is no RGB/sRGB output anywhere and no RGB teacher.
* Conditioning: SAVi's decoder consumes slots only.  N02 additionally conditions each
  candidate on the camera identity (learned embedding), the causal ``frame_index``
  (sinusoidal embedding, no fabricated seconds) and a per-camera global context projected
  from the encoder's validated multiscale features.  These are the declared "conditional
  measurement decoder" additions; they shift all candidates of one camera/time equally, so
  they cannot make an unusable candidate look better than another one.
* ``SaviPositionEmbedding`` supports the official ``linear`` and ``fourier`` embeddings with
  ``update_type="project_add"``; only the non-trainable form is ported because this decoder
  runs on several camera resolutions with one parameter set.  ``fourier`` is the default so
  the same module covers every resolution.
* The official ``options`` are kept (``early_fusion``); the target readout keeps the official
  ``Readout`` structure (one head module per target key) but the heads are 1x1-convolution
  MLPs of width ``hidden`` instead of a single ``Dense``, because this decoder emits several
  channel groups (mean / log-variance / four condition masks) per candidate.
* Variance is produced by the P0/P1 sensor layer (:func:`owac.frameworks.n02_full.sensor.
  softplus_variance`), so ``variance > 0`` always holds and the same ``log σ²`` term used by
  the frozen scoring path is differentiable; the decoder never dodges the residual by
  emitting an unbounded variance.

Measurement semantics: the decoder predicts measurements, it does not claim a full 3D
physical render.  Saturated queries are **censored** observations (``y >= level``): the
predicted value is never treated as an exact intensity, and
:func:`measurement_nll` scores them with the clip tail probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.frameworks.n02_full.encoder_v12 import LEVEL_IDS_V12, SpatialV12
from owac.frameworks.n02_full.sensor import (
    MAX_VARIANCE,
    MIN_VARIANCE,
    RawPrediction,
    clip_tail_log_prob,
    gaussian_nll,
    sanitise_variance,
    softplus_variance,
)

__all__ = [
    "MASK_NAMES_V12",
    "MeasurementQuery",
    "RawMeasurementDecoderV12",
    "SaviCNN",
    "SaviPositionEmbedding",
    "SaviReadout",
    "SpatialBroadcastDecoderV12",
    "convert_to_fourier_features",
    "create_gradient_grid",
    "measurement_nll",
    "spatial_broadcast",
]

#: Condition channels every decoder output carries, independently of the caller's masks.
MASK_NAMES_V12 = ("valid", "bad", "missing", "saturated")

#: Level used to define the decode grid: the finest direct spatial level.
_DECODE_LEVEL = LEVEL_IDS_V12[0]

#: Separator of the multi-camera ``"<camera>::<level>"`` key format.
_LEVEL_SEPARATOR = "::"


def _plain_level_dims(mapping: Mapping[str, int], *, name: str) -> dict[str, int]:
    """Accept ``{level: channels}`` or the composed ``{"<camera>::<level>": channels}`` keys.

    ``SpatialV12.level_dims`` uses the composed keys of a multi-camera pyramid while
    ``RawEncoderV12.spatial_dims`` uses the plain level IDs; both are valid configuration
    inputs for the decoder and the readout, so the composed keys are reduced here instead of
    failing later at forward time.
    """
    dims: dict[str, int] = {}
    for key, value in dict(mapping).items():
        level = str(key).split(_LEVEL_SEPARATOR)[-1]
        if level in dims and dims[level] != int(value):
            raise ValueError(f"{name} declares conflicting widths for level {level!r}")
        dims[level] = int(value)
    return dims


def _truncated_normal_(weight: torch.Tensor, *, std: float) -> None:
    """Flax/Haiku default initialiser: truncated normal with stddev ``std``.

    ``nn.Conv``/``nn.Dense`` defaults to ``lecun_normal`` (``VarianceScaling(scale=1,
    mode='fan_in', distribution='truncated_normal')``), i.e. stddev ``1/sqrt(fan_in)``, which
    is what every layer of the upstream port uses.  ``nn.init.trunc_normal_`` implements the
    same truncated-normal sampling as Flax's initialiser.
    """
    nn.init.trunc_normal_(weight, mean=0.0, std=float(std), a=-2.0, b=2.0)


# --------------------------------------------------------------------------------------
# savi/lib/utils.py ports
# --------------------------------------------------------------------------------------


def spatial_broadcast(x: torch.Tensor, resolution: Sequence[int]) -> torch.Tensor:
    """Broadcast flat inputs to a 2D grid (port of ``savi.lib.utils.spatial_broadcast``)."""
    height, width = int(resolution[0]), int(resolution[1])
    return x[:, None, None, :].expand(x.shape[0], height, width, x.shape[-1])


def create_gradient_grid(
    samples_per_dim: Sequence[int], value_range: Sequence[float] = (-1.0, 1.0)
) -> torch.Tensor:
    """Normalised coordinate grid (port of ``savi.lib.utils.create_gradient_grid``)."""
    axes = [
        torch.linspace(float(value_range[0]), float(value_range[1]), int(size))
        for size in samples_per_dim
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1)


def convert_to_fourier_features(inputs: torch.Tensor, basis_degree: int) -> torch.Tensor:
    """NeRF-style Fourier features (port of ``savi.lib.utils.convert_to_fourier_features``).

    ``inputs`` is ``(..., n_dims)`` with values in ``[-pi, pi]`` or ``[0, 2*pi]``; the output
    is ``(..., 2 * n_dims * basis_degree)`` holding ``[sin(x), sin(x + pi/2)]``.
    """
    if basis_degree < 1:
        raise ValueError("basis_degree must be >= 1")
    n_dims = inputs.shape[-1]
    identity = torch.eye(n_dims, dtype=inputs.dtype, device=inputs.device)
    freq_basis = torch.cat([(2.0**i) * identity for i in range(int(basis_degree))], dim=1)
    x = inputs @ freq_basis
    return torch.sin(torch.cat([x, x + 0.5 * math.pi], dim=-1))


# --------------------------------------------------------------------------------------
# savi/modules/misc.py port: PositionEmbedding
# --------------------------------------------------------------------------------------


class SaviPositionEmbedding(nn.Module):
    """Port of ``savi.modules.misc.PositionEmbedding`` (``project_add`` update type).

    ``embedding_type="linear"`` adds a learned projection of the normalised coordinate grid
    (2 channels) to the features; ``"fourier"`` uses the official NeRF-style Fourier basis
    (``2 * 2 * num_fourier_bases`` channels).  Both are resolution-free, so one module covers
    every camera resolution.  Only the non-trainable form (``trainable_pos_embedding=False``)
    is ported: upstream stores the grid as a non-trainable parameter and stops its gradient,
    which is exactly a constant given the input resolution.
    """

    def __init__(
        self,
        n_features: int,
        *,
        embedding_type: str = "fourier",
        update_type: str = "project_add",
        num_fourier_bases: int = 4,
    ):
        super().__init__()
        if embedding_type not in ("linear", "fourier"):
            raise ValueError(f"unknown embedding_type {embedding_type!r}")
        if update_type != "project_add":
            raise ValueError("only the official 'project_add' update type is ported")
        if embedding_type == "fourier" and num_fourier_bases < 1:
            raise ValueError("fourier position embeddings need at least one basis")
        self.n_features = int(n_features)
        self.embedding_type = embedding_type
        self.num_fourier_bases = int(num_fourier_bases)
        pos_dim = 2 if embedding_type == "linear" else 2 * 2 * self.num_fourier_bases
        self.pos_dim = int(pos_dim)
        self.dense_pe = nn.Linear(self.pos_dim, self.n_features)

    def _position_grid(self, height: int, width: int, *, device, dtype) -> torch.Tensor:
        grid = create_gradient_grid((height, width), (-1.0, 1.0)).to(device=device, dtype=dtype)
        if self.embedding_type == "fourier":
            return convert_to_fourier_features(grid * math.pi, self.num_fourier_bases)
        return grid

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = int(inputs.shape[-3]), int(inputs.shape[-2])
        position = self._position_grid(height, width, device=inputs.device, dtype=inputs.dtype)
        return inputs + self.dense_pe(position)


# --------------------------------------------------------------------------------------
# savi/modules/convolution.py port: CNN
# --------------------------------------------------------------------------------------


class SaviCNN(nn.Module):
    """Port of ``savi.modules.convolution.CNN`` (conv/transposed-conv stack, ReLU).

    Flax infers the input channel count lazily; this port takes it explicitly
    (``in_channels``), which is the only structural difference.
    """

    def __init__(
        self,
        in_channels: int,
        features: Sequence[int],
        kernel_size: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        layer_transpose: Sequence[bool],
        *,
        activation_fn: str = "relu",
        norm_type: str | None = None,
    ):
        super().__init__()
        features = tuple(int(item) for item in features)
        kernel_size = tuple(tuple(int(item) for item in pair) for pair in kernel_size)
        strides = tuple(tuple(int(item) for item in pair) for pair in strides)
        layer_transpose = tuple(bool(item) for item in layer_transpose)
        if not features:
            raise ValueError("the CNN needs at least one layer")
        if not (len(kernel_size) == len(strides) == len(layer_transpose) == len(features)):
            raise ValueError("features, kernel_size, strides and layer_transpose must match")
        if activation_fn != "relu":
            raise ValueError("only the official ReLU activation is ported")
        if norm_type not in (None, "group", "layer"):
            raise ValueError(f"unsupported norm_type {norm_type!r}")
        self.features = features
        self.kernel_size = kernel_size
        self.strides = strides
        self.layer_transpose = layer_transpose
        self.norm_type = norm_type
        self.activation_fn = activation_fn
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = int(in_channels)
        for out_channels, kernel, stride, transpose in zip(
            features, kernel_size, strides, layer_transpose, strict=True
        ):
            layer = (
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel, stride=stride, padding=1, output_padding=1
                )
                if transpose
                else nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=1)
            )
            self.layers.append(layer)
            norm: nn.Module | None = None
            if norm_type == "group":
                norm = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            elif norm_type == "layer":
                norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)
            self.norms.append(norm if norm is not None else nn.Identity())
            in_channels = out_channels
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for layer in self.layers:
            fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            _truncated_normal_(layer.weight, std=1.0 / math.sqrt(fan_in))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
        return x


# --------------------------------------------------------------------------------------
# savi/modules/misc.py port: Readout
# --------------------------------------------------------------------------------------


class SaviReadout(nn.Module):
    """Port of ``savi.modules.misc.Readout``: one head module per target key.

    ``stop_gradient`` is kept from the official module: a target whose flag is set is read
    from a detached copy of the backbone features (used here for the condition masks, which
    must not reshape the measurement field).
    """

    def __init__(
        self, heads: Mapping[str, nn.Module], *, stop_gradient: Mapping[str, bool] | None = None
    ):
        super().__init__()
        if not heads:
            raise ValueError("the readout needs at least one target")
        self.keys = tuple(str(key) for key in heads)
        stop_gradient = dict(stop_gradient or {})
        if set(stop_gradient) - set(self.keys):
            raise ValueError("stop_gradient names must be a subset of the target keys")
        self.heads = nn.ModuleDict({str(key): module for key, module in heads.items()})
        self.stop_gradient = {key: bool(stop_gradient.get(key, False)) for key in self.keys}

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for key in self.keys:
            source = inputs.detach() if self.stop_gradient[key] else inputs
            outputs[key] = self.heads[key](source)
        return outputs


class _HeadMLP(nn.Module):
    """1x1-convolution head with one hidden layer of width ``hidden`` (declared output layer)."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden, out_channels, kernel_size=1)
        _truncated_normal_(self.in_proj.weight, std=1.0 / math.sqrt(in_channels))
        nn.init.constant_(self.in_proj.bias, 0.0)
        _truncated_normal_(self.out_proj.weight, std=1.0 / math.sqrt(hidden))
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.gelu(self.in_proj(x)))


# --------------------------------------------------------------------------------------
# savi/modules/decoders.py port: SpatialBroadcastDecoder
# --------------------------------------------------------------------------------------


class SpatialBroadcastDecoderV12(nn.Module):
    """Port of ``savi.modules.decoders.SpatialBroadcastDecoder`` for measurement targets.

    ``forward(slots, resolution)`` takes ``(B, S, D)`` slots (candidate x slots already folded
    into the batch axis by the caller) and a ``(height, width)`` broadcast resolution, and
    returns per-target combined fields ``(B, H, W, C)`` plus the slot attention ``alphas``
    ``(B, S, H, W)`` for visualisation.
    """

    def __init__(
        self,
        slot_dim: int,
        target_channels: Mapping[str, int],
        *,
        hidden: int = 256,
        decoder_features: Sequence[int] | None = None,
        kernel_size: int = 3,
        num_fourier_bases: int = 4,
        early_fusion: bool = False,
    ):
        super().__init__()
        if not target_channels:
            raise ValueError("the decoder needs at least one measurement target")
        hidden = int(hidden)
        if decoder_features is None:
            # Official MoVi decoder width (64) at the declared hidden width 256.
            decoder_features = (max(16, hidden // 4),) * 3
        layer_transpose = (True,) * len(decoder_features)
        self.backbone = SaviCNN(
            int(slot_dim),
            features=decoder_features,
            kernel_size=((kernel_size, kernel_size),) * len(decoder_features),
            strides=((2, 2),) * len(decoder_features),
            layer_transpose=layer_transpose,
        )
        self.pos_emb = SaviPositionEmbedding(
            slot_dim,
            embedding_type="fourier",
            update_type="project_add",
            num_fourier_bases=num_fourier_bases,
        )
        self.hidden = hidden
        self.early_fusion = bool(early_fusion)
        self.target_channels = {str(key): int(value) for key, value in target_channels.items()}
        backbone_width = int(decoder_features[-1])
        self.alpha_logits = nn.Conv2d(backbone_width, 1, kernel_size=1)
        heads: dict[str, nn.Module] = {}
        for key, channels in self.target_channels.items():
            heads[key] = _HeadMLP(backbone_width, channels, hidden)
        self.target_readout = SaviReadout(
            heads, stop_gradient={"masks": True} if "masks" in heads else {}
        )
        _truncated_normal_(self.alpha_logits.weight, std=1.0 / math.sqrt(backbone_width))
        nn.init.constant_(self.alpha_logits.bias, 0.0)

    def forward(
        self, slots: torch.Tensor, resolution: Sequence[int]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if slots.ndim != 3:
            raise ValueError(f"slots must be (B, S, D), got {tuple(slots.shape)}")
        batch_size, n_slots, n_features = slots.shape
        if n_features != self.pos_emb.n_features:
            raise ValueError(
                f"slots carry {n_features} features but the decoder was built for "
                f"{self.pos_emb.n_features}"
            )
        x = slots.reshape(batch_size * n_slots, n_features)
        x = spatial_broadcast(x, resolution)
        x = self.pos_emb(x)  # (B*S, H, W, D)
        features = self.backbone(x.permute(0, 3, 1, 2).contiguous())  # (B*S, C, H, W)
        spatial_dims = tuple(int(item) for item in features.shape[-2:])

        alpha_logits = self.alpha_logits(features)
        # The official module keeps alpha on the last axis; the slot-axis softmax is identical.
        alpha_logits = alpha_logits.reshape(batch_size, n_slots, 1, *spatial_dims)
        alphas = torch.softmax(alpha_logits, dim=1)  # (B, S, 1, H, W)

        if self.early_fusion:
            fused = (features.reshape(batch_size, n_slots, -1, *spatial_dims) * alphas).sum(dim=1)
            targets = self.target_readout(fused)
            combined = {key: value for key, value in targets.items()}
        else:
            channels = self.target_readout(features)
            combined = {}
            for key, value in channels.items():
                reshaped = value.reshape(batch_size, n_slots, -1, *spatial_dims)
                combined[key] = (reshaped * alphas).sum(dim=1)  # (B, C, H, W)
        return combined, alphas.squeeze(2)


# --------------------------------------------------------------------------------------
# the frozen v1.2 decoder
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class MeasurementQuery:
    """One camera's sensor-space query set (declared units are the caller's)."""

    coords: torch.Tensor  # (B, Q, 2) float32, (row, column) sensor pixels
    camera_index: torch.Tensor  # (B, Q) int64, index into ``SpatialV12.camera_ids``
    value: torch.Tensor | None = None  # (B, Q) measured value, only when scoring
    masks: dict[str, torch.Tensor] | None = None  # data-side condition masks

    def __post_init__(self) -> None:
        if self.coords.ndim != 3 or self.coords.shape[-1] != 2:
            raise ValueError("query coords must be (B, Q, 2)")
        if self.camera_index.shape != self.coords.shape[:2]:
            raise ValueError("query camera_index must be (B, Q)")


class RawMeasurementDecoderV12(nn.Module):
    """Camera/time-conditioned RAW measurement decoder (SAVi spatial broadcast port).

    Args:
        in_dims: ``{level_id: channels}`` of the spatial features this decoder may consume
            (normally ``RawEncoderV12.spatial_dims``).
        slot_dim: width of one slot in ``slots``.
        n_slots: slots per candidate; a candidate composes its slots with the official
            alpha softmax.
        n_candidates: number of mutually exclusive candidate explanations (``K``).
        cam_embed/time_embed: widths of the learned camera embedding and of the sinusoidal
            ``frame_index`` embedding; both are projected into the slot width.
        hidden: width of the decoder's measurement heads (and the source of the decoder
            backbone width ``hidden // 4``).
        max_cameras: upper bound for the learned camera embedding table.
        measurement_channels: triplet channels the decoder predicts before the declared
            channel reduction; 3 for RAW3.
        channel_reduction: ``"mean"`` (scalar = co-sited triplet mean, the default) or
            ``"select"`` (requires the optional ``query_channel`` argument).
        saturation_level: declared clip level in sensor units; when set, saturated queries
            are censored at this level instead of being scored as exact intensities.
        units: declared measurement unit string used by :meth:`to_raw_prediction`.
    """

    def __init__(
        self,
        *,
        in_dims: Mapping[str, int] | None = None,
        level_dims: Mapping[str, int] | None = None,
        level_ids: Sequence[str] | None = None,
        camera_ids: Sequence[str] | None = None,
        d_model: int | None = None,
        slot_dim: int = 512,
        n_slots: int = 16,
        n_candidates: int = 4,
        cam_embed: int = 32,
        time_embed: int = 16,
        hidden: int = 256,
        max_cameras: int = 8,
        measurement_channels: int = 3,
        channel_reduction: str = "mean",
        saturation_level: float | None = None,
        units: str = "sensor_units",
        init_variance: float = 1.0,
        decoder_features: Sequence[int] | None = None,
        num_fourier_bases: int = 4,
    ):
        super().__init__()
        # ``in_dims`` is the frozen name; ``level_dims``/``level_ids``/``camera_ids``/``d_model``
        # are additive aliases the model integrator passes, and they never override the frozen
        # names.  Exactly one dimensional source may be given, and they must agree.
        if in_dims is not None and level_dims is not None:
            raise ValueError("pass either in_dims or its alias level_dims, not both")
        dims_source = in_dims if in_dims is not None else level_dims
        if dims_source is None:
            raise ValueError("in_dims (or its alias level_dims) must name the spatial levels")
        in_dims = _plain_level_dims(dims_source, name="in_dims")
        if not in_dims:
            raise ValueError("in_dims must name at least one spatial level")
        if level_ids is not None:
            unknown = [str(item) for item in level_ids if str(item) not in in_dims]
            if unknown:
                raise ValueError(f"level_ids names levels absent from in_dims: {unknown}")
            in_dims = {str(item): in_dims[str(item)] for item in level_ids}
        self.declared_camera_ids = (
            None if camera_ids is None else tuple(str(item) for item in camera_ids)
        )
        self.declared_d_model = None if d_model is None else int(d_model)
        if min(in_dims.values()) <= 0:
            raise ValueError("in_dims values must be positive channel counts")
        self.in_dims = in_dims
        self.slot_dim = int(slot_dim)
        self.n_slots = int(n_slots)
        self.n_candidates = int(n_candidates)
        self.cam_embed = int(cam_embed)
        self.time_embed = int(time_embed)
        self.hidden = int(hidden)
        self.max_cameras = int(max_cameras)
        if self.declared_camera_ids:
            self.max_cameras = max(self.max_cameras, len(self.declared_camera_ids))
        self.measurement_channels = int(measurement_channels)
        if channel_reduction not in ("mean", "select"):
            raise ValueError(f"unknown channel_reduction {channel_reduction!r}")
        self.channel_reduction = channel_reduction
        self.saturation_level = None if saturation_level is None else float(saturation_level)
        self.units = str(units)
        if not self.units:
            raise ValueError("units must be a non-empty declared unit string")
        if self.n_slots < 1 or self.n_candidates < 1:
            raise ValueError("n_slots and n_candidates must be positive")
        if self.time_embed < 2 or self.time_embed % 2:
            raise ValueError("time_embed must be an even number of sinusoidal dimensions")
        if self.measurement_channels < 1:
            raise ValueError("measurement_channels must be positive")

        # Observation context: per validated level mean -> slot width.
        self.context_proj = nn.ModuleDict(
            {level: nn.Linear(channels, self.slot_dim) for level, channels in in_dims.items()}
        )
        for module in self.context_proj.values():
            _truncated_normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            nn.init.constant_(module.bias, 0.0)

        self.camera_embed = nn.Embedding(max_cameras, cam_embed)
        _truncated_normal_(self.camera_embed.weight, std=1.0)
        self.cam_proj = nn.Linear(cam_embed, self.slot_dim)
        self.time_proj = nn.Linear(self.time_embed, self.slot_dim)
        for module in (self.cam_proj, self.time_proj):
            _truncated_normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            nn.init.constant_(module.bias, 0.0)

        target_channels = {
            "mean": self.measurement_channels,
            "log_variance": self.measurement_channels,
            "masks": len(MASK_NAMES_V12),
        }
        self.broadcast_decoder = SpatialBroadcastDecoderV12(
            self.slot_dim,
            target_channels,
            hidden=self.hidden,
            decoder_features=decoder_features,
            num_fourier_bases=num_fourier_bases,
        )
        if not math.isfinite(init_variance) or init_variance <= 0:
            raise ValueError("init_variance must be finite and positive")
        # Inverse softplus: log(v) would initialise variance to log(1 + v).
        variance_bias = init_variance if init_variance > 20 else math.log(math.expm1(init_variance))
        nn.init.constant_(
            self.broadcast_decoder.target_readout.heads["log_variance"].out_proj.bias, variance_bias
        )
        nn.init.constant_(self.broadcast_decoder.target_readout.heads["mean"].out_proj.bias, 0.0)

    # -- helpers ------------------------------------------------------------------------

    @staticmethod
    def _level_key(camera_id: str, level_id: str, multi: bool) -> str:
        return f"{camera_id}::{level_id}" if multi else level_id

    @staticmethod
    def _sinusoidal(values: torch.Tensor, dim: int) -> torch.Tensor:
        """Causal frame numbers (never seconds) -> sinusoidal features of width ``dim``."""
        half = dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=values.device)
            / max(half - 1, 1)
        )
        angles = values.to(torch.float32).unsqueeze(-1) * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _camera_context(self, features: SpatialV12, camera_index: int, batch: int) -> torch.Tensor:
        """Validated multiscale mean of one camera, projected to the slot width."""
        multi = len(features.camera_ids) > 1
        camera_id = features.camera_ids[camera_index]
        context = None
        for level_id, channels in self.in_dims.items():
            key = self._level_key(camera_id, level_id, multi)
            if key not in features.levels:
                continue
            values = features.levels[key]
            if int(values.shape[1]) != channels:
                raise ValueError(
                    f"level {key!r} carries {values.shape[1]} channels but in_dims declares "
                    f"{channels}"
                )
            valid = features.valid[key].to(values.dtype)
            count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
            pooled = (values * valid.unsqueeze(1)).sum(dim=(-1, -2)) / count.unsqueeze(-1)
            projected = self.context_proj[level_id](pooled)
            context = projected if context is None else context + projected
        if context is None:
            return torch.zeros(batch, self.slot_dim)
        return context / float(len(self.in_dims))

    @staticmethod
    def _sample(fields: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        """Sample ``(B, C, H, W)`` at ``grid`` ``(B, Q, 2)`` normalised coordinates."""
        sampled = F.grid_sample(
            fields,
            grid.unsqueeze(2),
            mode=mode,
            padding_mode="border",
            align_corners=False,
        )
        return sampled.squeeze(-1)

    def _sample_field(
        self,
        field: torch.Tensor,
        grid: torch.Tensor,
        *,
        batch: int,
        n_candidates: int,
        n_queries: int,
    ) -> torch.Tensor:
        """Sample one ``(B, K, C, H, W)`` field at ``grid`` -> ``(B, K, C, Q)``."""
        height, width = int(field.shape[-2]), int(field.shape[-1])
        flat = field.reshape(batch * n_candidates, -1, height, width)
        return self._sample(flat, grid).reshape(batch, n_candidates, -1, n_queries)

    def _query_grid(
        self, features: SpatialV12, camera_index: int, query_coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sensor coordinates -> normalised decode-grid coordinates for one camera.

        ``coords`` follows the P1 convention ``crop + packed_scale * (pad + cell + 0.5)``, so
        the inverse map is exact: ``cell = (coord - crop) / packed_scale`` and the padded grid
        covers ``[0, padded_shape]`` in frame cells.  ``grid_sample`` uses
        ``align_corners=False``, where a cell centre ``c + 0.5`` maps to ``2*(c+0.5)/N - 1``;
        with the level grid this is exactly the level cell centre used by the encoder.
        """
        camera_id = features.camera_ids[camera_index]
        transform = features.transforms.get(camera_id)
        if transform is None:
            raise ValueError(
                f"SpatialV12 carries no frame transform for camera {camera_id!r}; the decoder "
                "needs it to map sensor coordinates onto the decode grid"
            )
        crop = torch.tensor(
            transform.crop_offset, dtype=query_coords.dtype, device=query_coords.device
        )
        padded = torch.tensor(
            transform.padded_shape, dtype=query_coords.dtype, device=query_coords.device
        )
        cells = (query_coords - crop) / float(transform.packed_scale)
        normalised = 2.0 * cells / padded - 1.0
        inside = (normalised.abs() <= 1.0).all(dim=-1)  # (B, Q)
        rows = normalised[..., 0].clamp(-1.0, 1.0)
        columns = normalised[..., 1].clamp(-1.0, 1.0)
        # grid_sample wants (x, y); out-of-frame queries are clamped to the border and are
        # reported as not in support through ``inside``.
        return torch.stack((columns, rows), dim=-1), inside

    # -- forward ------------------------------------------------------------------------

    def forward(
        self,
        *,
        features: SpatialV12,
        slots: torch.Tensor,
        frame_index: torch.Tensor,
        query_coords: torch.Tensor,
        query_camera_index: torch.Tensor,
        query_channel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the measurement at every query point of every camera and candidate.

        Returns ``(mean, variance, masks)`` with ``mean``/``variance`` of shape ``(B, K, Q)``
        in sensor units and ``variance > 0`` (softplus of the predicted log-variance).
        ``masks`` holds ``valid/bad/missing/saturated`` (predicted probabilities, ``(B, K, Q)``),
        the data-side ``in_support`` (bool) and the censoring indicator ``saturated_censored``
        (bool, only when a ``saturation_level`` is declared).
        """
        if not isinstance(features, SpatialV12):
            raise TypeError(f"features must be a SpatialV12, got {type(features).__name__}")
        if slots.ndim == 3:
            slots = slots.unsqueeze(1)
        if slots.ndim != 4:
            raise ValueError(f"slots must be (B, K, N, D), got {tuple(slots.shape)}")
        batch, n_candidates, n_slots, slot_dim = slots.shape
        if n_candidates != self.n_candidates:
            raise ValueError(
                f"slots carry {n_candidates} candidates but the decoder was built for "
                f"{self.n_candidates}"
            )
        if n_slots != self.n_slots:
            raise ValueError(
                f"slots carry {n_slots} slots but the decoder was built for {self.n_slots}"
            )
        if slot_dim != self.slot_dim:
            raise ValueError(
                f"slots are {slot_dim} wide but the decoder was built for {self.slot_dim}"
            )
        if query_coords.shape[0] != batch or query_camera_index.shape != query_coords.shape[:2]:
            raise ValueError(
                "query_coords must be (B, Q, 2) and query_camera_index (B, Q) with the same B"
            )
        frame_index = torch.as_tensor(frame_index)
        if frame_index.ndim == 1 and frame_index.shape[0] != batch:
            raise ValueError(f"frame_index must carry one causal frame number per row ({batch})")
        if frame_index.ndim > 1:
            raise ValueError("frame_index must be (B,) or a scalar causal frame number")
        if bool((frame_index < 0).any()):
            raise ValueError("frame_index is a causal frame number and must be non-negative")
        n_cameras = len(features.camera_ids)
        if int(query_camera_index.min()) < 0 or int(query_camera_index.max()) >= n_cameras:
            raise ValueError("query_camera_index must index SpatialV12.camera_ids")
        if query_channel is not None:
            if query_channel.shape != query_coords.shape[:2]:
                raise ValueError("query_channel must be (B, Q) when provided")
            if self.channel_reduction == "select" and int(query_channel.max()) >= (
                self.measurement_channels
            ):
                raise ValueError("query_channel indexes measurement_channels")

        device = slots.device
        if frame_index.ndim == 0:
            frame_index = frame_index.expand(batch).to(device)
        else:
            frame_index = frame_index.to(device)
        time_features = self._sinusoidal(frame_index, self.time_embed)  # (B, T)
        time_term = self.time_proj(time_features).to(slots.dtype)  # (B, D)

        n_queries = int(query_coords.shape[1])
        mean = torch.zeros(batch, n_candidates, n_queries, device=device, dtype=torch.float32)
        variance = torch.zeros_like(mean)
        mask_probabilities = {name: torch.zeros_like(mean) for name in MASK_NAMES_V12}
        in_support = torch.zeros(batch, n_candidates, n_queries, device=device, dtype=torch.bool)
        common_layout = getattr(self, "group_camera_queries", False) and (
            batch == 1
            or torch.equal(query_camera_index, query_camera_index[:1].expand_as(query_camera_index))
        )

        for camera_index in range(n_cameras):
            camera_id = features.camera_ids[camera_index]
            selected = query_camera_index == camera_index  # (B, Q)
            if not bool(selected.any()):
                continue
            query_rows = (
                torch.nonzero(selected[0], as_tuple=False).flatten() if common_layout else None
            )
            local_coords = (
                query_coords if query_rows is None else query_coords.index_select(1, query_rows)
            )
            local_channels = (
                query_channel
                if query_rows is None or query_channel is None
                else query_channel.index_select(1, query_rows)
            )
            local_count = int(local_coords.shape[1])
            if camera_index >= self.max_cameras:
                raise ValueError(
                    f"camera index {camera_index} exceeds the declared max_cameras "
                    f"({self.max_cameras})"
                )
            multi = n_cameras > 1
            decode_key = self._level_key(camera_id, _DECODE_LEVEL, multi)
            if decode_key not in features.levels:
                raise ValueError(f"SpatialV12 carries no {_DECODE_LEVEL} level for {camera_id!r}")
            decode_reference = features.levels[decode_key]
            height, width = int(decode_reference.shape[-2]), int(decode_reference.shape[-1])

            # Condition the candidate slots on camera, causal frame number and the validated
            # multiscale context of this camera.  Every term is shared by all K candidates,
            # so it can shift the predicted measurement but never favour one candidate.
            camera_term = self.cam_proj(self.camera_embed.weight[camera_index]).to(slots.dtype)
            context_term = self._camera_context(features, camera_index, batch).to(slots.dtype)
            conditioned = slots + camera_term.view(1, 1, 1, -1) + time_term.view(batch, 1, 1, -1)
            conditioned = conditioned + context_term.view(batch, 1, 1, -1)

            upsampling = 2 ** len(self.broadcast_decoder.backbone.features)
            low_resolution = (
                max(1, math.ceil(height / upsampling)),
                max(1, math.ceil(width / upsampling)),
            )
            fields, _alphas = self.broadcast_decoder(
                conditioned.reshape(batch * n_candidates, n_slots, slot_dim), low_resolution
            )
            targets: dict[str, torch.Tensor] = {}
            for name, field in fields.items():
                if field.shape[-2:] != (height, width):
                    field = F.interpolate(
                        field, size=(height, width), mode="bilinear", align_corners=False
                    )
                targets[name] = field.reshape(
                    batch, n_candidates, field.shape[1], height, width
                ).to(torch.float32)

            grid, inside = self._query_grid(features, camera_index, local_coords)
            grid = (
                grid.unsqueeze(1)
                .expand(-1, n_candidates, -1, -1)
                .reshape(batch * n_candidates, local_count, 2)
            )

            sampled = {
                name: self._sample_field(
                    targets[name],
                    grid,
                    batch=batch,
                    n_candidates=n_candidates,
                    n_queries=local_count,
                )
                for name in ("mean", "log_variance", "masks")
            }
            mean_field = sampled["mean"]  # (B, K, C, Q)
            log_variance_field = sampled["log_variance"]
            mask_field = sampled["masks"]  # (B, K, 4, Q)
            if self.channel_reduction == "mean":
                channel_mean = mean_field.mean(dim=2)
                channel_log_variance = log_variance_field.mean(dim=2)
            else:
                if query_channel is None:
                    raise ValueError(
                        "channel_reduction='select' needs the optional query_channel argument"
                    )
                index = local_channels.to(device).view(batch, 1, 1, local_count)
                index = index.expand(batch, n_candidates, 1, local_count)
                channel_mean = torch.gather(mean_field, 2, index).squeeze(2)
                channel_log_variance = torch.gather(log_variance_field, 2, index).squeeze(2)

            camera_variance = softplus_variance(
                channel_log_variance, min_variance=MIN_VARIANCE, max_variance=MAX_VARIANCE
            )
            mask_probs = torch.sigmoid(mask_field)
            # Dense per-camera sampling: keep the queries of this camera and drop the rest.
            keep = selected.view(batch, 1, n_queries)

            def merge(existing, values, query_rows=query_rows, keep=keep):
                return (
                    existing.index_copy(2, query_rows, values)
                    if query_rows is not None
                    else torch.where(keep, values, existing)
                )

            mean = merge(mean, channel_mean)
            variance = merge(variance, camera_variance)
            for position, name in enumerate(MASK_NAMES_V12):
                mask_probabilities[name] = merge(
                    mask_probabilities[name], mask_probs[:, :, position]
                )
            # Data-side support of the query point: the encoder's own validity map.
            reference_valid = features.valid[decode_key]
            supports = self._sample(
                reference_valid.to(torch.float32)
                .unsqueeze(1)
                .expand(batch, n_candidates, -1, -1)
                .reshape(batch * n_candidates, 1, height, width),
                grid,
                mode="nearest",
            ).reshape(batch, n_candidates, 1, local_count)
            in_support = merge(
                in_support,
                (supports.squeeze(2) > 0.5) & inside.unsqueeze(1).expand(-1, n_candidates, -1),
            )

        masks: dict[str, torch.Tensor] = dict(mask_probabilities)
        masks["in_support"] = in_support
        if self.saturation_level is not None:
            masks["saturated_censored"] = mean >= float(self.saturation_level)
        return mean, variance, masks

    # -- scoring helpers ----------------------------------------------------------------

    def to_raw_prediction(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        masks: Mapping[str, torch.Tensor] | None = None,
        *,
        units: str | None = None,
    ) -> RawPrediction:
        """Frozen :class:`~owac.frameworks.n02_full.sensor.RawPrediction` view of the output.

        The shapes ``(B, K, Q)`` are kept as they are; ``variance_components`` splits the
        declared sensor floor from the learned model variance so a caller can list them
        separately.  The caller owns the unit declaration; no implicit re-scaling happens.
        """
        # The decoder already returns softplus variances; only the sensor clamp is applied
        # here so the units and the values are exactly the ones that were scored.
        variance = sanitise_variance(variance, min_variance=MIN_VARIANCE, max_variance=MAX_VARIANCE)
        return RawPrediction(
            mean=mean,
            variance=variance,
            masks=dict(masks or {}),
            units=str(units or self.units),
            variance_components={
                "sensor": torch.full_like(variance, MIN_VARIANCE),
                "model": (variance - MIN_VARIANCE).clamp_min(0.0),
                "total": variance,
            },
        )

    def nll(
        self,
        target: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        *,
        masks: Mapping[str, torch.Tensor] | None = None,
        saturation_level: float | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Masked ``log σ²`` measurement NLL of :func:`measurement_nll` for this model."""
        level = self.saturation_level if saturation_level is None else saturation_level
        return measurement_nll(
            target,
            mean,
            variance,
            masks=masks,
            saturation_level=level,
            reduction=reduction,
        )


def measurement_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    masks: Mapping[str, torch.Tensor] | None = None,
    saturation_level: float | None = None,
    saturation_side: str = "top",
    reduce_masks: Sequence[str] = ("in_support", "valid"),
    exclude_masks: Sequence[str] = ("bad", "missing"),
    mask_threshold: float = 0.5,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
    reduction: str = "mean",
) -> torch.Tensor:
    """Gaussian measurement NLL with the ``log σ²`` term, masks and saturation censoring.

    The contribution of every scored entry is
    ``0.5*((y - mu)^2 / v + log v + log 2 pi)`` (``v`` clamped by the sensor layer, so an
    unbounded variance is not a way to escape the residual).  A saturated entry is a
    **censored** observation: when ``saturation_level`` is declared it is scored by the clip
    tail probability ``-log P(y >= level)`` instead, and when no level is declared it is
    excluded from the loss ("explicitly not scored") rather than treated as an exact
    intensity.  Non-finite targets and clamped-out entries are excluded as well.

    Args:
        masks: data-side condition masks (produced by the data layer, not the decoder's own
            predictions); float masks are thresholded at ``mask_threshold``.
        reduce_masks: masks that must hold for an entry to be scored.
        exclude_masks: masks that drop an entry.
        reduction: ``"mean"`` over scored entries, ``"sum"``, or ``"none"`` for the
            per-entry tensor (unsatisfied entries are then exactly zero).
    """
    if target.shape != mean.shape or mean.shape != variance.shape:
        raise ValueError(
            f"target/mean/variance must share a shape, got {tuple(target.shape)}, "
            f"{tuple(mean.shape)}, {tuple(variance.shape)}"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"unknown reduction {reduction!r}")

    def _as_bool(name: str) -> torch.Tensor | None:
        if not masks or name not in masks:
            return None
        value = masks[name]
        if value.dtype == torch.bool:
            return value
        return value > float(mask_threshold)

    weights = torch.isfinite(target)
    for name in reduce_masks:
        flag = _as_bool(name)
        if flag is not None:
            weights = weights & flag
    for name in exclude_masks:
        flag = _as_bool(name)
        if flag is not None:
            weights = weights & ~flag
    saturated = _as_bool("saturated")
    censored = torch.zeros_like(weights)
    if saturated is not None:
        if saturation_level is not None:
            censored = saturated & weights
        else:
            weights = weights & ~saturated

    values = target.to(mean.dtype)
    nll = gaussian_nll(values, mean, variance, min_variance=min_variance, max_variance=max_variance)
    if bool(censored.any()):
        tail = -clip_tail_log_prob(
            mean,
            variance,
            level=float(saturation_level),
            side=saturation_side,
            min_variance=min_variance,
            max_variance=max_variance,
        )
        nll = torch.where(censored, tail, nll)
    weights = weights | censored
    scored = nll * weights.to(nll.dtype)
    if reduction == "sum":
        return scored.sum()
    if reduction == "none":
        return scored
    count = weights.to(nll.dtype).sum().clamp_min(1.0)
    return scored.sum() / count


def decoder_summary(decoder: RawMeasurementDecoderV12) -> dict[str, Any]:
    """Compact configuration record for reports (no tensors, no randomness)."""
    return {
        "in_dims": dict(decoder.in_dims),
        "slot_dim": decoder.slot_dim,
        "n_slots": decoder.n_slots,
        "n_candidates": decoder.n_candidates,
        "cam_embed": decoder.cam_embed,
        "time_embed": decoder.time_embed,
        "hidden": decoder.hidden,
        "measurement_channels": decoder.measurement_channels,
        "channel_reduction": decoder.channel_reduction,
        "saturation_level": decoder.saturation_level,
        "units": decoder.units,
        "parameters": sum(parameter.numel() for parameter in decoder.parameters()),
    }

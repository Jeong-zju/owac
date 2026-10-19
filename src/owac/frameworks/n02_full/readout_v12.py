"""N02 v1.2 policy readout: Perceiver IO port.

PyTorch port of the official Perceiver IO core (``perceiver.py``) and its position
encoding (``position_encoding.py``)::

    repository google-deepmind/deepmind-research @ f5de0ede8430809180254ee957abf36ed62579ef
    perceiver/perceiver.py            Attention/attend, MLP, SelfAttention, CrossAttention,
                                      PerceiverEncoder, BasicDecoder
    perceiver/position_encoding.py    generate_fourier_features, build_linear_positions,
                                      FourierPositionEncoding, PositionEncodingProjector
    paper: Perceiver IO: a general architecture for structured inputs & outputs
           (https://arxiv.org/abs/2107.14795), Jaegle et al. 2021, Apache-2.0 code.

``owac.frameworks.n02.readout`` is the *old token readout* of the v0.1 line and is **not**
this module; nothing from it is reused here.

Port differences (declared, not hidden):

* Flax/Haiku infers channel counts lazily; every attention module here takes its
  ``qk_channels``/``v_channels``/``output_channels`` explicitly, and matrices use
  ``nn.Linear`` where upstream uses ``hk.Linear`` (``conv_1d``).  The attention math is the
  official one: ``softmax(q k^T / sqrt(head_dim)) v`` with the official 1e30 mask fill and
  the official "wipe fully masked query rows" rule.
* ``PerceiverEncoder`` keeps the official structure (one cross-attend, then
  ``num_blocks`` self-attend blocks sharing weights via a loop over blocks) and this port
  maps the readout's ``depth`` onto ``num_blocks`` with one self-attend per block.
* The official ``Perceiver`` input side is replaced by the N02 token builder: the readout
  consumes (a) the direct multi-scale RAW spatial features with their sensor coordinates and
  camera identity, (b) the corrected latent state (candidate slots) and (c) the candidate
  support ``q``.  Candidates stay separate tokens; ``log q`` enters the encoder
  cross-attention as an additive key bias (this is the declared support weighting) and the
  per-candidate attention mass is reported in ``aux``.  Candidate tokens carry no candidate
  index embedding, so candidate order cannot become a label.
* Queries are the official decoder queries (Fourier position encoding of a fixed per-camera
  index grid, optionally projected by ``PositionEncodingProjector``) plus a trainable query
  embedding and a learned camera embedding, so each camera's queries carry camera identity.
* ``N02ReadoutV12`` returns ``(B, n_cameras * n_queries, out_dim)`` and exposes the real
  attention tensors for visualisation.

Attention tensors for visualisation (one sentence each, as required):

* ``aux["attention"]``: decoder cross-attention weights ``(B, Q_total, n_latents)``, softmax
  over the latent axis (already normalized per query), averaged over heads.
* ``aux["encoder_attention"]``: encoder cross-attention weights ``(B, n_latents, T)``,
  softmax over the input-token axis, averaged over heads.
* ``aux["candidate_attention"]``: mass of ``encoder_attention`` on each candidate's tokens,
  renormalized over the K candidates (sums to 1), i.e. how much the latents read each
  candidate explanation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from owac.frameworks.n02_full.encoder_v12 import SpatialV12

__all__ = [
    "N02ReadoutV12",
    "PerceiverAttention",
    "PerceiverCrossAttention",
    "PerceiverDecoder",
    "PerceiverEncoder",
    "PerceiverMLP",
    "PerceiverSelfAttention",
    "attend",
    "build_linear_positions",
    "generate_fourier_features",
]

_MASK_FILL = 1e30

#: Separator of the multi-camera ``"<camera>::<level>"`` key format.
_LEVEL_SEPARATOR = "::"


def _plain_level_dims(mapping: Mapping[str, int]) -> dict[str, int]:
    """Accept ``{level: channels}`` or the composed ``{"<camera>::<level>": channels}`` keys.

    ``SpatialV12.level_dims`` uses composed keys for a multi-camera pyramid and
    ``RawEncoderV12.spatial_dims`` uses plain level IDs; both are valid, so the composed keys
    are reduced here instead of failing at forward time.
    """
    dims: dict[str, int] = {}
    for key, value in dict(mapping).items():
        level = str(key).split(_LEVEL_SEPARATOR)[-1]
        if level in dims and dims[level] != int(value):
            raise ValueError(f"spatial_dims declares conflicting widths for level {level!r}")
        dims[level] = int(value)
    return dims


def _truncated_normal_(weight: torch.Tensor, *, std: float) -> None:
    """Haiku ``VarianceScaling``/``TruncatedNormal`` initialiser (stddev ``std``)."""
    nn.init.trunc_normal_(weight, mean=0.0, std=float(std), a=-2.0, b=2.0)


# --------------------------------------------------------------------------------------
# perceiver/perceiver.py: primitives
# --------------------------------------------------------------------------------------


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_prob: float = 0.0,
    attention_mask: torch.Tensor | None = None,
    attention_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-head attention (port of ``perceiver.attend``), also returning the weights.

    Args:
        q/k/v: ``(B, T, H, D)`` query / key / value.
        attention_mask: ``(B, q_indices, kv_indices)`` bool, True where attention is valid.
        attention_bias: optional ``(B, kv_indices)`` additive key bias (N02 extension used for
            the candidate support weight ``log q``).
    """
    batch, q_indices, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attention = torch.einsum("bthd,bThd->bhtT", q, k)
    attention = attention * (1.0 / math.sqrt(q_head_dim))
    if attention_bias is not None:
        attention = attention + attention_bias[:, None, None, :]
    if attention_mask is not None:
        attention = torch.where(
            attention_mask[:, None, :, :],
            attention,
            torch.full_like(attention, -_MASK_FILL),
        )
    normalized = torch.softmax(attention, dim=-1)
    if dropout_prob > 0.0:
        normalized = torch.nn.functional.dropout(normalized, p=dropout_prob, training=True)
    summed = torch.einsum("bhtT,bThd->bthd", normalized, v)
    summed = summed.reshape(batch, q_indices, hiddens)
    if attention_mask is not None:
        wipe = torch.all(~attention_mask, dim=2, keepdim=True)  # (B, T, 1)
        summed = torch.where(wipe, torch.zeros_like(summed), summed)
    return summed, normalized


class PerceiverAttention(nn.Module):
    """Port of ``perceiver.Attention`` (multi-headed {cross, self}-attention)."""

    def __init__(
        self,
        *,
        qk_channels: int,
        v_channels: int,
        output_channels: int,
        num_heads: int = 8,
        init_scale: float = 1.0,
        with_final_bias: bool = True,
        final_init_scale_multiplier: float = 1.0,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if qk_channels % num_heads or v_channels % num_heads:
            raise ValueError("qk_channels and v_channels must be divisible by num_heads")
        self.qk_channels = int(qk_channels)
        self.v_channels = int(v_channels)
        self.output_channels = int(output_channels)
        self.num_heads = int(num_heads)
        self.dropout_prob = float(dropout_prob)
        self._q = nn.Linear(qk_channels, qk_channels)
        self._k = nn.Linear(qk_channels, qk_channels)
        self._v = nn.Linear(qk_channels, v_channels)
        self._out = nn.Linear(v_channels, output_channels, bias=with_final_bias)
        for layer, scale in (
            (self._q, init_scale),
            (self._k, init_scale),
            (self._v, init_scale),
            (self._out, final_init_scale_multiplier * init_scale),
        ):
            _truncated_normal_(layer.weight, std=scale / math.sqrt(layer.in_features))
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._q(inputs_q)
        k = self._k(inputs_kv)
        v = self._v(inputs_kv)
        batch, q_time, _ = q.shape
        kv_time = k.shape[1]
        q = q.reshape(batch, q_time, self.num_heads, self.qk_channels // self.num_heads)
        k = k.reshape(batch, kv_time, self.num_heads, self.qk_channels // self.num_heads)
        v = v.reshape(batch, kv_time, self.num_heads, self.v_channels // self.num_heads)
        result, weights = attend(
            q,
            k,
            v,
            dropout_prob=self.dropout_prob,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )
        return self._out(result), weights


class PerceiverMLP(nn.Module):
    """Port of ``perceiver.MLP`` (one hidden layer with GELU)."""

    def __init__(self, channels: int, *, widening_factor: int = 4, init_scale: float = 1.0):
        super().__init__()
        self.input = nn.Linear(channels, widening_factor * channels)
        self.output = nn.Linear(widening_factor * channels, channels)
        for layer in (self.input, self.output):
            _truncated_normal_(layer.weight, std=init_scale / math.sqrt(layer.in_features))
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(nn.functional.gelu(self.input(x)))


class PerceiverSelfAttention(nn.Module):
    """Port of ``perceiver.SelfAttention`` (pre-norm attention + pre-norm MLP, residuals)."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 8,
        widening_factor: int = 1,
        att_init_scale: float = 1.0,
        dense_init_scale: float = 1.0,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.norm_att = nn.LayerNorm(channels)
        self.attention = PerceiverAttention(
            qk_channels=channels,
            v_channels=channels,
            output_channels=channels,
            num_heads=num_heads,
            init_scale=att_init_scale,
            dropout_prob=dropout_prob,
        )
        self.norm_mlp = nn.LayerNorm(channels)
        self.mlp = PerceiverMLP(
            channels, widening_factor=widening_factor, init_scale=dense_init_scale
        )

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normalized = self.norm_att(inputs)
        attention, _ = self.attention(normalized, normalized, attention_mask)
        x = inputs + attention
        return x + self.mlp(self.norm_mlp(x))


class PerceiverCrossAttention(nn.Module):
    """Port of ``perceiver.CrossAttention`` (pre-norm q/kv, query residual, MLP block)."""

    def __init__(
        self,
        *,
        q_channels: int,
        kv_channels: int,
        output_channels: int | None = None,
        qk_channels: int | None = None,
        v_channels: int | None = None,
        num_heads: int = 8,
        widening_factor: int = 1,
        att_init_scale: float = 1.0,
        dense_init_scale: float = 1.0,
        use_query_residual: bool = True,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        output_channels = int(output_channels or q_channels)
        self.norm_q = nn.LayerNorm(q_channels)
        self.norm_kv = nn.LayerNorm(kv_channels)
        self.attention = PerceiverAttention(
            qk_channels=int(qk_channels or kv_channels),
            v_channels=int(v_channels or qk_channels or kv_channels),
            output_channels=output_channels,
            num_heads=num_heads,
            init_scale=att_init_scale,
            dropout_prob=dropout_prob,
        )
        self.use_query_residual = bool(use_query_residual)
        self.norm_mlp = nn.LayerNorm(output_channels)
        self.mlp = PerceiverMLP(
            output_channels, widening_factor=widening_factor, init_scale=dense_init_scale
        )

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention, weights = self.attention(
            self.norm_q(inputs_q),
            self.norm_kv(inputs_kv),
            attention_mask,
            attention_bias,
        )
        x = inputs_q + attention if self.use_query_residual else attention
        return x + self.mlp(self.norm_mlp(x)), weights


def make_cross_attention_mask(query_mask: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
    """Port of ``perceiver.make_cross_attention_mask``."""
    if query_mask.shape[0] != kv_mask.shape[0]:
        raise ValueError("query_mask and kv_mask must share the batch axis")
    return torch.einsum("bq,bk->bqk", query_mask.to(torch.bool), kv_mask.to(torch.bool))


# --------------------------------------------------------------------------------------
# perceiver/position_encoding.py
# --------------------------------------------------------------------------------------


def build_linear_positions(
    index_dims: Sequence[int],
    output_range: Sequence[float] = (-1.0, 1.0),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Port of ``position_encoding.build_linear_positions`` (``(..., N)`` index grid).

    ``device``/``dtype`` are additive: upstream builds the grid on the default device, this
    port places it where the consuming linear layer lives.
    """
    axes = [
        torch.linspace(
            float(output_range[0]),
            float(output_range[1]),
            int(size),
            device=device,
            dtype=dtype,
        )
        for size in index_dims
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1)


def generate_fourier_features(
    pos: torch.Tensor,
    num_bands: int,
    max_resolution: Sequence[int] = (224, 224),
    concat_pos: bool = True,
    sine_only: bool = False,
) -> torch.Tensor:
    """Port of ``position_encoding.generate_fourier_features``.

    ``pos`` is ``(n, d)`` in ``[-1, 1]``; the frequency bands run linearly from 1 to
    ``resolution / 2`` per dimension, exactly as upstream.
    """
    min_freq = 1.0
    if num_bands < 1:
        raise ValueError("num_bands must be >= 1")
    if len(max_resolution) != pos.shape[-1]:
        raise ValueError("max_resolution must have one entry per position dimension")
    freq_bands = torch.stack(
        [torch.linspace(min_freq, float(res) / 2.0, int(num_bands)) for res in max_resolution],
        dim=0,
    ).to(pos.device)
    per_pos = pos[:, :, None] * freq_bands[None, :, :]
    per_pos = per_pos.reshape(pos.shape[0], -1)
    if sine_only:
        per_pos = torch.sin(math.pi * per_pos)
    else:
        per_pos = torch.cat([torch.sin(math.pi * per_pos), torch.cos(math.pi * per_pos)], dim=-1)
    if concat_pos:
        per_pos = torch.cat([pos, per_pos], dim=-1)
    return per_pos


class FourierPositionEncoding(nn.Module):
    """Port of ``position_encoding.FourierPositionEncoding``."""

    def __init__(
        self,
        index_dims: Sequence[int],
        num_bands: int,
        *,
        concat_pos: bool = True,
        max_resolution: Sequence[int] | None = None,
        sine_only: bool = False,
    ):
        super().__init__()
        self.index_dims = tuple(int(item) for item in index_dims)
        self.num_bands = int(num_bands)
        self.concat_pos = bool(concat_pos)
        self.sine_only = bool(sine_only)
        self.max_resolution = tuple(int(item) for item in (max_resolution or self.index_dims))

    def forward(
        self,
        batch_size: int,
        pos: torch.Tensor | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if pos is None:
            grid = build_linear_positions(self.index_dims, device=device, dtype=dtype)
            positions = grid.reshape(-1, grid.shape[-1])
            positions = positions.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if pos.ndim != 3:
                raise ValueError("pos must be (batch, n, d)")
            positions = pos
        n_positions = int(positions.shape[1])
        features = generate_fourier_features(
            positions.reshape(-1, positions.shape[-1]),
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        return features.reshape(batch_size, n_positions, features.shape[-1])


class PositionEncodingProjector(nn.Module):
    """Port of ``position_encoding.PositionEncodingProjector`` (linear projection)."""

    def __init__(self, output_size: int, base_position_encoding: FourierPositionEncoding):
        super().__init__()
        self.base_position_encoding = base_position_encoding
        probe = base_position_encoding(1)
        self.linear = nn.Linear(probe.shape[-1], int(output_size))
        _truncated_normal_(self.linear.weight, std=1.0 / math.sqrt(self.linear.in_features))
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, batch_size: int, pos: torch.Tensor | None = None) -> torch.Tensor:
        # The index grid is built where the projection lives so the module works on any device.
        return self.linear(
            self.base_position_encoding(
                batch_size,
                pos,
                device=self.linear.weight.device,
                dtype=self.linear.weight.dtype,
            )
        )


# --------------------------------------------------------------------------------------
# perceiver/perceiver.py: PerceiverEncoder / BasicDecoder
# --------------------------------------------------------------------------------------


class PerceiverEncoder(nn.Module):
    """Port of ``perceiver.PerceiverEncoder`` (trainable latents, one cross-attend, blocks)."""

    def __init__(
        self,
        *,
        num_self_attends_per_block: int = 6,
        num_blocks: int = 8,
        z_index_dim: int = 512,
        num_z_channels: int = 1024,
        num_cross_attend_heads: int = 1,
        num_self_attend_heads: int = 8,
        cross_attend_widening_factor: int = 1,
        self_attend_widening_factor: int = 1,
        dropout_prob: float = 0.0,
        z_pos_enc_init_scale: float = 0.02,
    ):
        super().__init__()
        if num_z_channels % num_self_attend_heads or num_z_channels % num_cross_attend_heads:
            raise ValueError("num_z_channels must be divisible by both head counts")
        self.num_blocks = int(num_blocks)
        self.num_self_attends_per_block = int(num_self_attends_per_block)
        self.z_index_dim = int(z_index_dim)
        self.num_z_channels = int(num_z_channels)
        # Official TrainablePositionEncoding(z_index_dim, num_z_channels, 0.02).
        self.z_pos_enc = nn.Parameter(torch.zeros(self.z_index_dim, self.num_z_channels))
        _truncated_normal_(self.z_pos_enc, std=z_pos_enc_init_scale)
        self.cross_attend = PerceiverCrossAttention(
            q_channels=num_z_channels,
            kv_channels=num_z_channels,
            output_channels=num_z_channels,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            dropout_prob=dropout_prob,
            use_query_residual=True,
        )
        self.self_attends = nn.ModuleList(
            [
                PerceiverSelfAttention(
                    num_z_channels,
                    num_heads=num_self_attend_heads,
                    widening_factor=self_attend_widening_factor,
                    dropout_prob=dropout_prob,
                )
                for _ in range(self.num_self_attends_per_block)
            ]
        )

    def latents(self, batch_size: int) -> torch.Tensor:
        return self.z_pos_enc.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(
        self,
        inputs: torch.Tensor,
        z: torch.Tensor,
        *,
        input_mask: torch.Tensor | None = None,
        input_attention_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                torch.ones(z.shape[:2], dtype=torch.bool, device=z.device), input_mask
            )
        z, cross_weights = self.cross_attend(
            z, inputs, attention_mask=attention_mask, attention_bias=input_attention_bias
        )
        for _ in range(self.num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z)
        return z, cross_weights


class PerceiverDecoder(nn.Module):
    """Port of ``perceiver.BasicDecoder`` (cross-attention decoder with final projection)."""

    def __init__(
        self,
        *,
        output_num_channels: int,
        output_index_dims: Sequence[int],
        num_z_channels: int,
        num_heads: int = 1,
        num_bands: int = 32,
        use_query_residual: bool = False,
        final_project: bool = True,
    ):
        super().__init__()
        self.output_num_channels = int(output_num_channels)
        self.output_index_dims = tuple(int(item) for item in output_index_dims)
        self.num_bands = int(num_bands)
        self.output_pos_enc = PositionEncodingProjector(
            num_z_channels,
            FourierPositionEncoding(index_dims=self.output_index_dims, num_bands=self.num_bands),
        )
        self.cross_attention = PerceiverCrossAttention(
            q_channels=num_z_channels,
            kv_channels=num_z_channels,
            output_channels=num_z_channels,
            num_heads=num_heads,
            widening_factor=1,
            use_query_residual=use_query_residual,
        )
        self.final_project = bool(final_project)
        self.final_layer = (
            nn.Linear(num_z_channels, self.output_num_channels) if self.final_project else None
        )
        if self.final_layer is not None:
            _truncated_normal_(
                self.final_layer.weight, std=1.0 / math.sqrt(self.final_layer.in_features)
            )
            nn.init.constant_(self.final_layer.bias, 0.0)

    def decoder_query(self, batch_size: int, pos: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_pos_enc(batch_size, pos)

    def forward(
        self,
        query: torch.Tensor,
        z: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask, torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
            )
        output, weights = self.cross_attention(query, z, attention_mask=attention_mask)
        if self.final_layer is not None:
            output = self.final_layer(output)
        return output, weights


# --------------------------------------------------------------------------------------
# the frozen v1.2 readout
# --------------------------------------------------------------------------------------


class N02ReadoutV12(nn.Module):
    """Perceiver IO readout consuming direct RAW features, candidate state and support.

    Args:
        spatial_dims: ``{level_id: channels}`` of the spatial levels to consume (normally
            ``RawEncoderV12.spatial_dims``).
        latent_dim: width of the latent array and of every input token.
        n_latents: number of latents (official default in the Perceiver paper's ImageNet
            setting is 512; the v1.2 starting configuration uses 128).
        n_queries: queries **per camera**; the returned token count is
            ``n_cameras * n_queries``.
        out_dim: output token width (2048 = the π0.5 hidden width).
        num_heads: attention heads.
        depth: number of encoder self-attention blocks after the initial cross-attend.
        max_cameras: upper bound of the learned camera embedding table.
        num_bands: Fourier bands of the query/input position encodings.
        widening_factor: encoder MLP widening factor (official encoder default is 1).
    """

    N_QUERIES = 256
    OUT_DIM = 2048

    def __init__(
        self,
        *,
        spatial_dims: Mapping[str, int] | None = None,
        level_dims: Mapping[str, int] | None = None,
        camera_ids: Sequence[str] | None = None,
        d_model: int | None = None,
        latent_dim: int | None = 512,
        n_latents: int = 128,
        n_queries: int = 256,
        out_dim: int = 2048,
        num_heads: int = 8,
        depth: int = 2,
        max_cameras: int = 8,
        num_bands: int = 32,
        widening_factor: int = 1,
        state_dim: int = 512,
        input_max_resolution: Sequence[int] = (64, 64),
        preserve_spatial_tokens: bool = False,
        output_token_rms: float = 4.0,
        context_gate_init: float = 0.1,
        scale_balance: bool = False,
        scale_balance_mode: str = "contrast",
        learned_global_fusion: bool = False,
    ):
        super().__init__()
        # ``spatial_dims`` is the frozen name; ``level_dims``/``camera_ids``/``d_model`` are
        # additive aliases the model integrator passes and never override the frozen names.
        if spatial_dims is not None and level_dims is not None:
            raise ValueError("pass either spatial_dims or its alias level_dims, not both")
        dims_source = spatial_dims if spatial_dims is not None else level_dims
        if dims_source is None:
            raise ValueError("spatial_dims (or its alias level_dims) must name the spatial levels")
        spatial_dims = _plain_level_dims(dims_source)
        if not spatial_dims:
            raise ValueError("spatial_dims must name at least one level")
        self.spatial_dims = spatial_dims
        self.level_ids = tuple(spatial_dims)
        self.declared_camera_ids = (
            None if camera_ids is None else tuple(str(item) for item in camera_ids)
        )
        # Frozen default stays 512; the declared d_model alias is only the default *source*.
        self.latent_dim = (
            int(d_model)
            if (latent_dim is None and d_model is not None)
            else int(latent_dim if latent_dim is not None else 512)
        )
        self.n_latents = int(n_latents)
        self.n_queries = int(n_queries)
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.depth = int(depth)
        self.max_cameras = int(max_cameras)
        if self.declared_camera_ids:
            self.max_cameras = max(self.max_cameras, len(self.declared_camera_ids))
        self.num_bands = int(num_bands)

        self.level_proj = nn.ModuleDict(
            {
                level: nn.Linear(channels, self.latent_dim)
                for level, channels in spatial_dims.items()
            }
        )
        self.level_embed = nn.Parameter(torch.zeros(len(self.level_ids), self.latent_dim))
        _truncated_normal_(self.level_embed, std=0.02)
        self.camera_embed = nn.Embedding(self.max_cameras, self.latent_dim)
        _truncated_normal_(self.camera_embed.weight, std=0.02)
        for module in self.level_proj.values():
            _truncated_normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            nn.init.constant_(module.bias, 0.0)

        # Input positions are normalised cell centres; the declared reference resolution is
        # the 256x256 start frame's stride-4 grid (64 cells), so the Fourier bands run from
        # 1 to its Nyquist frequency 32.  One encoding serves every camera.
        self.input_max_resolution = tuple(int(item) for item in input_max_resolution)
        self.input_pos_enc = PositionEncodingProjector(
            self.latent_dim,
            FourierPositionEncoding(
                index_dims=self.input_max_resolution,
                num_bands=self.num_bands,
                max_resolution=self.input_max_resolution,
            ),
        )

        self.state_proj = nn.Linear(int(state_dim), self.latent_dim)
        self.support_proj = nn.Linear(1, self.latent_dim)
        for module in (self.state_proj, self.support_proj):
            _truncated_normal_(module.weight, std=1.0 / math.sqrt(module.in_features))
            nn.init.constant_(module.bias, 0.0)

        self.encoder = PerceiverEncoder(
            num_self_attends_per_block=1,
            num_blocks=max(1, self.depth),
            z_index_dim=self.n_latents,
            num_z_channels=self.latent_dim,
            num_cross_attend_heads=self.num_heads,
            num_self_attend_heads=self.num_heads,
            cross_attend_widening_factor=widening_factor,
            self_attend_widening_factor=widening_factor,
        )
        grid = int(round(math.isqrt(self.n_queries)))
        self.query_grid = (grid, grid) if grid * grid == self.n_queries else (self.n_queries,)
        self.decoder = PerceiverDecoder(
            output_num_channels=self.out_dim,
            output_index_dims=self.query_grid,
            num_z_channels=self.latent_dim,
            num_heads=self.num_heads,
            num_bands=self.num_bands,
            use_query_residual=False,
            final_project=True,
        )
        self.query_embed = nn.Parameter(torch.zeros(1, self.n_queries, self.latent_dim))
        _truncated_normal_(self.query_embed, std=0.02)
        self.preserve_spatial_tokens = bool(preserve_spatial_tokens)
        self.scale_balance = bool(scale_balance)
        if scale_balance_mode not in ("contrast", "rms"):
            raise ValueError("scale_balance_mode must be contrast or rms")
        self.scale_balance_mode = scale_balance_mode
        self.learned_global_fusion = bool(learned_global_fusion)
        if self.learned_global_fusion and not (self.scale_balance and self.preserve_spatial_tokens):
            raise ValueError("global/local fusion requires the balanced spatial branch")
        if self.scale_balance and not self.preserve_spatial_tokens:
            raise ValueError("balanced fusion requires the direct spatial branch")
        self.output_token_rms = float(output_token_rms)
        if self.preserve_spatial_tokens:
            if len(self.query_grid) != 2 or not self.output_token_rms > 0:
                raise ValueError("spatial readout requires a square query grid and positive RMS")
            if not 0 < context_gate_init < 1:
                raise ValueError("context_gate_init must lie in (0, 1)")
            self.spatial_skip = nn.Linear(sum(self.spatial_dims.values()), self.out_dim, bias=False)
            nn.init.xavier_uniform_(self.spatial_skip.weight)
            self.context_gate = nn.Parameter(torch.tensor(math.atanh(context_gate_init)))
            if self.learned_global_fusion:
                self.scale_logits = nn.Parameter(torch.zeros(len(self.level_ids)))
                self.local_gain = nn.Parameter(torch.tensor(1.0))
                self.global_gain = nn.Parameter(torch.tensor(0.1))

    def spatial_output_tokens(self, features: SpatialV12) -> torch.Tensor:
        """Sample each camera's actual frame at fixed local positions, bypassing latent pooling.

        All scales use sensor coordinates and the original frame extent, so bottom/right
        padding is not stretched into the image. Invalid measurements are zero before
        interpolation. No RGB image or teacher is constructed.
        """
        side = self.query_grid[0]
        groups = []
        multi = len(features.camera_ids) > 1
        for camera in features.camera_ids:
            transform = features.transforms[camera]
            first = features.levels[self._level_key(camera, self.level_ids[0], multi)]
            batch = first.shape[0]
            rows = torch.arange(side, device=first.device, dtype=first.dtype) + 0.5
            yy, xx = torch.meshgrid(rows, rows, indexing="ij")
            sensor = (
                torch.stack(
                    [
                        yy * transform.frame_shape[0] / side * transform.packed_scale
                        + transform.crop_offset[0],
                        xx * transform.frame_shape[1] / side * transform.packed_scale
                        + transform.crop_offset[1],
                    ],
                    dim=-1,
                )
                .unsqueeze(0)
                .expand(batch, -1, -1, -1)
            )
            normal = self._normalised_coords(sensor, camera, features)
            grid = normal[..., [1, 0]]
            samples = []
            sample_valids = []
            for level in self.level_ids:
                key = self._level_key(camera, level, multi)
                value = features.levels[key]
                valid = features.valid[key].to(value.dtype).reshape(batch, 1, *value.shape[-2:])
                mass = torch.nn.functional.grid_sample(valid, grid, align_corners=False)
                sampled = torch.nn.functional.grid_sample(
                    value.masked_fill(valid == 0, 0), grid, align_corners=False
                ) / mass.clamp_min(1e-6)
                samples.append(sampled.flatten(2).transpose(1, 2))
                sample_valids.append(mass.flatten(2).squeeze(1) > 1e-6)
            if self.scale_balance:
                from owac.frameworks.n02_full.representation_v12 import scale_rms, spatial_contrast

                normalize = scale_rms if self.scale_balance_mode == "rms" else spatial_contrast

                parts = []
                globals_ = []
                offset = 0
                for sample, sample_valid in zip(samples, sample_valids, strict=True):
                    width = sample.shape[-1]
                    conditioned = normalize(sample, sample_valid)
                    projected = torch.nn.functional.linear(
                        conditioned, self.spatial_skip.weight[:, offset : offset + width]
                    )
                    parts.append(normalize(projected, sample_valid))
                    if self.learned_global_fusion:
                        # Preserve each image's channel-mean vector in a separate
                        # path. RMS controls units without removing image content.
                        conditioned_global = scale_rms(sample, sample_valid)
                        mass = sample_valid.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
                        pooled = conditioned_global.sum(1, keepdim=True) / mass
                        globals_.append(
                            torch.nn.functional.linear(
                                pooled, self.spatial_skip.weight[:, offset : offset + width]
                            )
                        )
                    offset += width
                if self.learned_global_fusion:
                    weights = self.scale_logits.softmax(0) * math.sqrt(len(parts))
                    local = (torch.stack(parts) * weights[:, None, None, None]).sum(0)
                    global_ = (torch.stack(globals_) * weights[:, None, None, None]).sum(0)
                else:
                    local = torch.stack(parts).sum(0) / math.sqrt(len(parts))
            else:
                local = self.spatial_skip(torch.cat(samples, dim=-1))
            local = torch.nn.functional.layer_norm(local, (self.out_dim,))
            if self.learned_global_fusion:
                global_ = torch.nn.functional.layer_norm(global_, (self.out_dim,))
                local = self.local_gain * local + self.global_gain * global_
            groups.append(local * self.output_token_rms)
        return torch.cat(groups, dim=1)

    # -- input construction -------------------------------------------------------------

    def _level_key(self, camera_id: str, level_id: str, multi: bool) -> str:
        return f"{camera_id}::{level_id}" if multi else level_id

    def _normalised_coords(
        self, coords: torch.Tensor, camera_id: str, features: SpatialV12
    ) -> torch.Tensor:
        """Sensor coordinates -> ``[-1, 1]`` positions (frame cell centres)."""
        transform = features.transforms.get(camera_id)
        if transform is None:
            raise ValueError(f"SpatialV12 carries no frame transform for camera {camera_id!r}")
        crop = torch.tensor(transform.crop_offset, dtype=coords.dtype, device=coords.device)
        padded = torch.tensor(transform.padded_shape, dtype=coords.dtype, device=coords.device)
        cells = (coords - crop) / float(transform.packed_scale)
        return (2.0 * cells / padded - 1.0).clamp(-1.0, 1.0)

    def build_tokens(
        self,
        features: SpatialV12,
        state_tokens: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Assemble the encoder input tokens ``(B, T, latent_dim)`` and their bookkeeping.

        Order: every camera's spatial levels (finest first), then the ``K * N`` candidate
        state tokens.  ``log q`` is *not* baked into the values; it is returned as the
        additive key bias of the encoder cross-attention.
        """
        if not isinstance(features, SpatialV12):
            raise TypeError(f"features must be a SpatialV12, got {type(features).__name__}")
        if state_tokens.ndim == 3:
            state_tokens = state_tokens.unsqueeze(1)
        if state_tokens.ndim != 4:
            raise ValueError(f"state_tokens must be (B, K, N, D), got {tuple(state_tokens.shape)}")
        batch, n_candidates, n_slots, state_dim = state_tokens.shape
        if state_dim != self.state_proj.in_features:
            raise ValueError(
                f"state_tokens are {state_dim} wide but the readout was built for "
                f"{self.state_proj.in_features}"
            )
        if support.ndim != 2 or support.shape != (batch, n_candidates):
            raise ValueError(
                f"support must be (B, K) = {(batch, n_candidates)}, got {tuple(support.shape)}"
            )
        multi = len(features.camera_ids) > 1

        tokens: list[torch.Tensor] = []
        camera_index: list[int] = []
        level_index: list[int] = []
        for position, camera_id in enumerate(features.camera_ids):
            if position >= self.max_cameras:
                raise ValueError(
                    f"camera index {position} exceeds the declared max_cameras ({self.max_cameras})"
                )
            camera_term = self.camera_embed.weight[position].to(state_tokens.dtype)
            for level_position, level_id in enumerate(self.level_ids):
                key = self._level_key(camera_id, level_id, multi)
                if key not in features.levels:
                    raise ValueError(f"SpatialV12 carries no {level_id!r} level for {camera_id!r}")
                values = features.levels[key]
                coords = features.coords[key]
                if int(values.shape[1]) != self.spatial_dims[level_id]:
                    raise ValueError(
                        f"level {key!r} carries {values.shape[1]} channels but spatial_dims "
                        f"declares {self.spatial_dims[level_id]}"
                    )
                height, width = int(values.shape[-2]), int(values.shape[-1])
                flat = values.reshape(batch, values.shape[1], height * width).permute(0, 2, 1)
                if self.scale_balance:
                    valid_flat = features.valid[key].reshape(batch, height * width)
                    flat = flat.masked_fill(~valid_flat.unsqueeze(-1).bool(), 0)
                    flat = torch.nn.functional.layer_norm(flat, (flat.shape[-1],))
                projected = self.level_proj[level_id](flat)
                positions = self._normalised_coords(
                    coords.reshape(batch, height * width, 2), camera_id, features
                )
                positional = self.input_pos_enc(batch, positions)
                projected = projected + positional
                projected = projected + self.level_embed[level_position].view(1, 1, -1)
                projected = projected + camera_term.view(1, 1, -1)
                tokens.append(projected)
                camera_index.extend([position] * (height * width))
                level_index.extend([level_position] * (height * width))

        support_feature = self.support_proj(
            support.to(state_tokens.dtype).reshape(batch, n_candidates, 1, 1)
        )
        state = self.state_proj(state_tokens)
        state = state + support_feature  # q as a declared feature, no candidate index
        state_tokens_flat = state.reshape(batch, n_candidates * n_slots, self.latent_dim)
        tokens.append(state_tokens_flat)
        n_spatial = sum(item.shape[1] for item in tokens[:-1])
        token_camera = torch.tensor(
            camera_index + [-1] * (n_candidates * n_slots), dtype=torch.long
        )
        token_level = torch.tensor(level_index + [-1] * (n_candidates * n_slots), dtype=torch.long)
        token_candidate = torch.tensor(
            [-1] * n_spatial + [index for index in range(n_candidates) for _ in range(n_slots)],
            dtype=torch.long,
        )
        input_tokens = torch.cat(tokens, dim=1)
        bookkeeping = {
            "token_camera_index": token_camera,
            "token_level_index": token_level,
            "token_candidate_index": token_candidate,
            "n_spatial_tokens": torch.tensor(n_spatial),
            "n_candidates": torch.tensor(n_candidates),
            "n_slots": torch.tensor(n_slots),
        }
        return input_tokens, token_candidate, bookkeeping

    # -- forward ------------------------------------------------------------------------

    def forward(
        self,
        *,
        features: SpatialV12,
        state_tokens: torch.Tensor,
        support: torch.Tensor,
        masks: Mapping[str, Any] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Read out ``(B, n_cameras * n_queries, out_dim)`` tokens plus attention ``aux``."""
        input_tokens, token_candidate, bookkeeping = self.build_tokens(
            features, state_tokens, support
        )
        batch, n_tokens, _ = input_tokens.shape

        if masks is None:
            spatial_valid = self._spatial_valid(features)
            state_valid = torch.ones(
                batch,
                n_tokens - spatial_valid.shape[1],
                dtype=torch.bool,
                device=input_tokens.device,
            )
            input_valid = torch.cat([spatial_valid, state_valid], dim=1)
        elif isinstance(masks, torch.Tensor):
            input_valid = masks.to(torch.bool)
        elif "input_valid" in masks:
            input_valid = torch.as_tensor(masks["input_valid"]).to(torch.bool)
        else:
            raise ValueError(
                "masks must be None, a bool (B, T) tensor, or a mapping with 'input_valid'"
            )
        if input_valid.shape != (batch, n_tokens):
            raise ValueError(f"input_valid must be (B, T) = {(batch, n_tokens)}")

        # Declared support weighting: log q is an additive key bias of the encoder
        # cross-attention.  Candidate tokens receive it; spatial tokens receive nothing.
        candidate_ids = token_candidate.to(input_tokens.device)
        is_candidate = candidate_ids.view(1, -1) >= 0
        support_flat = support.to(input_tokens.dtype).unsqueeze(1).expand(-1, n_tokens, -1)
        index = candidate_ids.clamp_min(0).view(1, -1, 1).expand(batch, -1, 1)
        gathered = torch.gather(support_flat, 2, index).squeeze(-1)
        if (
            not bool(torch.isfinite(support).all())
            or bool((support < 0).any())
            or bool((support.sum(dim=-1) <= 0).any())
        ):
            raise ValueError("support must be finite, nonnegative, and have positive row mass")
        input_valid = input_valid.to(input_tokens.device) & (~is_candidate | (gathered > 0))
        bias = torch.where(
            is_candidate,
            torch.log(gathered.clamp_min(1e-12)),
            torch.zeros((), dtype=input_tokens.dtype, device=input_tokens.device),
        )

        latents = self.encoder.latents(batch)
        z, encoder_weights = self.encoder(
            input_tokens, latents, input_mask=input_valid, input_attention_bias=bias
        )

        query = self.decoder.decoder_query(batch) + self.query_embed.expand(batch, -1, -1)
        queries: list[torch.Tensor] = []
        for position in range(len(features.camera_ids)):
            queries.append(query + self.camera_embed.weight[position].view(1, 1, -1))
        decoder_query = torch.cat(queries, dim=1)

        tokens_out, decoder_weights = self.decoder(decoder_query, z)
        if self.preserve_spatial_tokens:
            local = self.spatial_output_tokens(features)
            context = torch.nn.functional.layer_norm(tokens_out, (self.out_dim,))
            tokens_out = local + self.context_gate.tanh() * self.output_token_rms * context

        # Head-averaged attention summaries for visualisation.
        attention = decoder_weights.mean(dim=1)  # (B, Q_total, n_latents)
        encoder_attention = encoder_weights.mean(dim=1)  # (B, n_latents, T)
        candidate_mass = (
            encoder_attention * is_candidate.to(encoder_attention.dtype).view(1, 1, -1)
        ).sum(dim=1)  # (B, T)
        n_candidates = int(bookkeeping["n_candidates"])
        n_slots = int(bookkeeping["n_slots"])
        n_spatial = int(bookkeeping["n_spatial_tokens"])
        candidate_attention = (
            candidate_mass[:, n_spatial:].reshape(batch, n_candidates, n_slots).sum(dim=2)
        )
        candidate_attention = candidate_attention / candidate_attention.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)

        aux: dict[str, torch.Tensor] = {
            "attention": attention,
            "encoder_attention": encoder_attention,
            "candidate_attention": candidate_attention,
            "latents": z,
            "token_valid": input_valid,
            "support": support,
        }
        aux.update(bookkeeping)
        return tokens_out, aux

    def _spatial_valid(self, features: SpatialV12) -> torch.Tensor:
        """``(B, T_spatial)`` validity of the spatial tokens in :meth:`build_tokens` order."""
        multi = len(features.camera_ids) > 1
        flags: list[torch.Tensor] = []
        for camera_id in features.camera_ids:
            for level_id in self.level_ids:
                key = self._level_key(camera_id, level_id, multi)
                valid = features.valid[key]
                flags.append(valid.reshape(valid.shape[0], -1))
        return torch.cat(flags, dim=1)

    # -- reporting ----------------------------------------------------------------------

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_summary(self) -> dict[str, Any]:
        return {
            "spatial_dims": dict(self.spatial_dims),
            "latent_dim": self.latent_dim,
            "n_latents": self.n_latents,
            "n_queries": self.n_queries,
            "out_dim": self.out_dim,
            "num_heads": self.num_heads,
            "depth": self.depth,
            "learned_global_fusion": self.learned_global_fusion,
            "query_grid": list(self.query_grid),
            "input_max_resolution": list(self.input_max_resolution),
            "preserve_spatial_tokens": self.preserve_spatial_tokens,
            "scale_balance": self.scale_balance,
            "scale_balance_mode": self.scale_balance_mode,
            "parameters": self.parameter_count(),
        }

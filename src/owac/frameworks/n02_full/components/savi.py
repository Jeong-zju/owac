"""SAVi / SAVi++ operator port (R02), JAX/Flax -> PyTorch.

Upstream (verbatim copies kept next to this file)::

    components/upstream/savi/savi/modules/attention.py
    components/upstream/savi/savi/modules/video.py
    components/upstream/savi/savi/modules/misc.py
    components/upstream/savi/savi/modules/decoders.py
    components/upstream/savi/savi/modules/initializers.py
    repository google-research/slot-attention-video @ ba8f15ee19472c6f9425c9647daf87910f17b605

Reference environment note (measured; see
``outputs/n02-full/p0/parity/logs/30_savi_jax_export.log``):
upstream ``misc.GRU`` calls ``nn.GRUCell()`` with no argument, but flax >= 0.7 made ``features`` a
required field.  The reference run therefore executes the upstream modules under the installed
flax (0.10.4) with ``upstream_refs.patch_flax_grucell`` restoring the defaulted field; the cell
math and the parameter names are unchanged between flax 0.6.11 and 0.10.4 (both implement
``r = sigmoid(ir(x) + hr(h))``, ``z = sigmoid(iz(x) + hz(h))``,
``n = tanh(in(x) + r * hn(h))``, ``h' = (1 - z) n + z h``).  Installing flax 0.6.11 instead was
tried and failed: it needs an older jax (``jax.config.define_bool_state`` is missing in the
environment's jax 0.4.35 and 0.4.26).

Port differences against upstream (also in ``third_party.lock.json``):

* Flax -> torch parameter layout. ``nn.Dense``/``nn.DenseGeneral`` kernels are ``(in, out)``;
  they are transposed into torch ``Linear`` weights on load. Flax ``LayerNorm`` has ``scale`` and
  ``bias`` (``use_bias=True`` default) and reduces in float32; both are kept.
* No randomness: the upstream modules used here (SlotAttention/GRU/MLP/attention) contain no
  dropout, so eval-mode torch matches JAX exactly.
* ``Processor`` reproduces the ``nn.scan`` order (corrector first, then predictor; the carried
  state is the *predicted* state) but runs a Python loop over time instead of ``lax.scan``: it is
  numerically equivalent and not compiled/parallel in time.
* **Extension, not upstream**: :class:`SlotAttentionMasked` implements the effective-mask path
  (invalid keys genuinely masked out, softmax over the slot axis, renormalised by the valid input
  mass). Upstream ``SlotAttention.__call__`` discards ``padding_mask`` entirely and
  ``GeneralizedDotProductAttention`` raises ``NotImplementedError`` for any attention bias, so
  this is new behaviour with its own tests and must not be reported as upstream parity.

``general_dense_q_0`` / ``general_dense_k_0`` / ``general_dense_v_0`` and the GRU gate names
(``ir``/``hr``/``iz``/``hz``/``in``/``hn``) follow the upstream Flax names so a Flax parameter
pytree can be loaded by path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

UPSTREAM_REPOSITORY = "google-research/slot-attention-video"
UPSTREAM_COMMIT = "ba8f15ee19472c6f9425c9647daf87910f17b605"
REFERENCE_FLAX_VERSION = "0.10.4+GRUCellCompat"


def flatten_flax_params(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a Flax parameter pytree into ``{"a/b/kernel": array}`` style paths."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_flax_params(value, path))
        else:
            flat[path] = value
    return flat


def _tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _dense_from_flax(flat: Mapping[str, Any], prefix: str, bias: bool = True) -> nn.Linear:
    kernel = flat[f"{prefix}/kernel"]
    out_features, in_features = kernel.shape[1], kernel.shape[0]
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.copy_(_tensor(kernel).t().contiguous())
        if bias:
            layer.bias.copy_(_tensor(flat[f"{prefix}/bias"]).reshape(-1))
    return layer


class FlaxLayerNorm(nn.Module):
    """Flax ``nn.LayerNorm`` (last-axis normalisation, ``scale`` and ``bias``)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.features = features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        var = ((x32 - mean) ** 2).mean(dim=-1, keepdim=True)
        out = (x32 - mean) / torch.sqrt(var + self.eps)
        out = out * self.scale.float() + self.bias.float()
        return out.to(dtype)

    def load_flax(self, flat: Mapping[str, Any], prefix: str) -> None:
        with torch.no_grad():
            self.scale.copy_(_tensor(flat[f"{prefix}/scale"]).reshape(-1))
            self.bias.copy_(_tensor(flat[f"{prefix}/bias"]).reshape(-1))


class FlaxDenseGeneral(nn.Module):
    """Flax ``nn.DenseGeneral(axis=-1, features=(num_heads, head_dim), use_bias=False)``."""

    def __init__(self, in_features: int, num_heads: int, head_dim: int):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.empty(num_heads * head_dim, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("...i,oi->...o", x, self.weight)
        return out.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def load_flax(self, flat: Mapping[str, Any], prefix: str) -> None:
        # Flax keeps the feature tuple in the kernel shape: (in_features, heads, head_dim).
        kernel = _tensor(flat[f"{prefix}/kernel"])
        with torch.no_grad():
            self.weight.copy_(kernel.reshape(kernel.shape[0], -1).t().contiguous())


class GRUCell(nn.Module):
    """Flax ``nn.GRUCell`` (0.6.11 semantics: reset applied *after* the hidden projection).

    ``r = sigmoid(ir(x) + hr(h))``, ``z = sigmoid(iz(x) + hz(h))``,
    ``n = tanh(in(x) + r * hn(h))``, ``h' = (1 - z) * n + z * h``.
    Gate order is (r, z, n) and the reset multiplies the hidden pre-activation, not ``h``.
    """

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.ir = nn.Linear(in_features, hidden_features)
        self.hr = nn.Linear(hidden_features, hidden_features, bias=False)
        self.iz = nn.Linear(in_features, hidden_features)
        self.hz = nn.Linear(hidden_features, hidden_features, bias=False)
        self.in_ = nn.Linear(in_features, hidden_features)
        self.hn = nn.Linear(hidden_features, hidden_features)

    def forward(self, carry: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.ir(inputs) + self.hr(carry))
        z = torch.sigmoid(self.iz(inputs) + self.hz(carry))
        n = torch.tanh(self.in_(inputs) + r * self.hn(carry))
        return (1.0 - z) * n + z * carry

    def load_flax(self, flat: Mapping[str, Any], prefix: str) -> None:
        for name, layer, bias in (
            ("ir", self.ir, True),
            ("hr", self.hr, False),
            ("iz", self.iz, True),
            ("hz", self.hz, False),
            ("in", self.in_, True),
            ("hn", self.hn, True),
        ):
            _load_dense(layer, flat, f"{prefix}/{name}", bias)


def _load_dense(layer: nn.Linear, flat: Mapping[str, Any], prefix: str, bias: bool) -> None:
    kernel = flat[f"{prefix}/kernel"]
    with torch.no_grad():
        layer.weight.copy_(_tensor(kernel).t().contiguous())
        if bias:
            layer.bias.copy_(_tensor(flat[f"{prefix}/bias"]).reshape(-1))


class MLP(nn.Module):
    """``savi.modules.misc.MLP`` with one or more hidden layers and optional pre-LayerNorm."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int = 1,
        activation: str = "relu",
        layernorm: str | None = None,
        activate_output: bool = False,
        residual: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.layernorm_mode = layernorm
        self.activate_output = activate_output
        self.residual = residual
        self.layernorm: FlaxLayerNorm | None = None
        self.hidden_layers = nn.ModuleList()
        self.output_layer: nn.Linear | None = None

    def build(self, input_size: int) -> MLP:
        """Flax modules get their shapes from the input; call this before loading parameters."""
        self.input_size = input_size
        if self.layernorm_mode == "pre":
            self.layernorm = FlaxLayerNorm(input_size)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(input_size if i == 0 else self.hidden_size, self.hidden_size)
                for i in range(self.num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(
            self.hidden_size if self.num_hidden_layers else input_size, self.output_size
        )
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        if self.layernorm is not None:
            x = self.layernorm(x)
        for layer in self.hidden_layers:
            x = _activate(layer(x), self.activation)
        assert self.output_layer is not None, "call MLP.build(input_size) first"
        x = self.output_layer(x)
        if self.activate_output:
            x = _activate(x, self.activation)
        if self.residual:
            x = x + inputs
        return x

    def load_flax(self, flat: Mapping[str, Any], prefix: str) -> None:
        if self.layernorm is not None:
            self.layernorm.load_flax(flat, f"{prefix}/LayerNorm_0")
        for i, layer in enumerate(self.hidden_layers):
            _load_dense(layer, flat, f"{prefix}/dense_mlp_{i}", True)
        assert self.output_layer is not None
        _load_dense(self.output_layer, flat, f"{prefix}/dense_mlp_{self.num_hidden_layers}", True)


def _activate(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "relu":
        return torch.relu(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    if name == "tanh":
        return torch.tanh(x)
    raise KeyError(f"unsupported activation {name!r}")


class InvertedDotProductAttention(nn.Module):
    """``savi.modules.attention.InvertedDotProductAttention``.

    Softmax runs over the *query* axis (slot axis) and, with ``renormalize_keys`` (norm_type
    "mean"), the attention weights are divided by their key-axis sum plus epsilon, i.e. value
    aggregation becomes a weighted mean instead of a sum.
    """

    def __init__(
        self,
        num_heads: int,
        multi_head: bool = False,
        norm_type: str | None = "mean",
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.multi_head = multi_head
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.out_proj: FlaxDenseGeneral | None = None
        self.layernorm: FlaxLayerNorm | None = None
        self.value_features: int | None = None

    def build(self, value_features: int) -> InvertedDotProductAttention:
        self.value_features = value_features * (1 if self.multi_head else 1)
        if self.multi_head:
            self.out_proj = FlaxDenseGeneral(value_features * self.num_heads, 1, value_features)
        if self.norm_type == "layernorm":
            self.layernorm = FlaxLayerNorm(value_features)
        return self

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Shapes: query/key ``(..., q, heads, d)``, value ``(..., k, heads, dv)``."""
        scale = float(query.shape[-1]) ** 0.5
        attn = torch.einsum("...qhd,...khd->...hqk", query / scale, key)
        attn = torch.softmax(attn, dim=-2)  # query (slot) axis
        if self.norm_type == "mean":
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        out = torch.einsum("...hqk,...khd->...qhd", attn, value)
        if self.multi_head:
            assert self.out_proj is not None
            out = self.out_proj(out.reshape(*out.shape[:-2], -1))
            out = out.squeeze(-2)
        else:
            out = out.squeeze(-2)
        if self.norm_type == "layernorm" and self.layernorm is not None:
            out = self.layernorm(out)
        return out


class SlotAttention(nn.Module):
    """``savi.modules.attention.SlotAttention`` (upstream, no mask path)."""

    def __init__(
        self,
        num_iterations: int = 1,
        qkv_size: int | None = None,
        mlp_size: int | None = None,
        epsilon: float = 1e-8,
        num_heads: int = 1,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.input_norm: FlaxLayerNorm | None = None
        self.q_norm: FlaxLayerNorm | None = None
        self.dense_q: FlaxDenseGeneral | None = None
        self.dense_k: FlaxDenseGeneral | None = None
        self.dense_v: FlaxDenseGeneral | None = None
        self.attention: InvertedDotProductAttention | None = None
        self.gru: GRUCell | None = None
        self.mlp: MLP | None = None

    def build(self, slot_size: int, inputs_size: int) -> SlotAttention:
        self.slot_size = slot_size
        qkv_size = self.qkv_size or slot_size
        if qkv_size % self.num_heads:
            raise ValueError("qkv_size must be divisible by num_heads")
        head_dim = qkv_size // self.num_heads
        self.qkv_size = qkv_size
        self.input_norm = FlaxLayerNorm(inputs_size)
        self.q_norm = FlaxLayerNorm(slot_size)
        self.dense_q = FlaxDenseGeneral(slot_size, self.num_heads, head_dim)
        self.dense_k = FlaxDenseGeneral(inputs_size, self.num_heads, head_dim)
        self.dense_v = FlaxDenseGeneral(inputs_size, self.num_heads, head_dim)
        self.attention = InvertedDotProductAttention(
            self.num_heads, multi_head=self.num_heads > 1, norm_type="mean", epsilon=1e-8
        ).build(head_dim)
        self.gru = GRUCell(qkv_size, slot_size)
        if self.mlp_size is not None:
            self.mlp = MLP(
                hidden_size=self.mlp_size,
                output_size=slot_size,
                num_hidden_layers=1,
                activation="relu",
                layernorm="pre",
                residual=True,
            ).build(slot_size)
        return self

    def forward(self, slots: torch.Tensor, inputs: torch.Tensor, **_: Any) -> torch.Tensor:
        assert self.input_norm is not None and self.gru is not None
        assert self.dense_k is not None and self.dense_v is not None
        assert self.dense_q is not None and self.q_norm is not None
        assert self.attention is not None
        inputs_n = self.input_norm(inputs)
        k = self.dense_k(inputs_n)
        v = self.dense_v(inputs_n)
        for _ in range(self.num_iterations):
            q = self.dense_q(self.q_norm(slots))
            updates = self.attention(q, k, v)
            slots = self.gru(slots, updates)
            if self.mlp is not None:
                slots = self.mlp(slots)
        return slots

    def load_flax(self, flat: Mapping[str, Any], prefix: str) -> None:
        """Load a Flax parameter pytree flattened by :func:`flatten_flax_params`."""
        assert self.input_norm and self.q_norm and self.dense_q and self.dense_k
        assert self.dense_v and self.attention and self.gru
        # Flax numbers modules in *construction* order: upstream constructs `layernorm_q` before
        # the input LayerNorm, so LayerNorm_0 is the query norm and LayerNorm_1 the input norm.
        self.q_norm.load_flax(flat, f"{prefix}/LayerNorm_0")
        self.input_norm.load_flax(flat, f"{prefix}/LayerNorm_1")
        self.dense_q.load_flax(flat, f"{prefix}/general_dense_q_0")
        self.dense_k.load_flax(flat, f"{prefix}/general_dense_k_0")
        self.dense_v.load_flax(flat, f"{prefix}/general_dense_v_0")
        self.gru.load_flax(flat, f"{prefix}/GRU_0/GRUCell_0")
        if self.attention.multi_head and self.attention.out_proj is not None:
            self.attention.out_proj.load_flax(
                flat, f"{prefix}/InvertedDotProductAttention_0/DenseGeneral_0"
            )
        if self.attention.layernorm is not None:
            self.attention.layernorm.load_flax(
                flat, f"{prefix}/InvertedDotProductAttention_0/LayerNorm_0"
            )
        if self.mlp is not None:
            self.mlp.load_flax(flat, f"{prefix}/MLP_0")


class SlotAttentionMasked(SlotAttention):
    """**Extension (not upstream)**: SlotAttention that really uses an input validity mask.

    Invalid keys are removed from both the softmax normalisation and the value aggregation, the
    softmax still runs over the slot axis, and the aggregated update is renormalised by the valid
    input mass so that a slot seeing no valid input receives a zero update instead of an
    arbitrary distribution over pad tokens.
    """

    def forward(  # type: ignore[override]
        self,
        slots: torch.Tensor,
        inputs: torch.Tensor,
        valid: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if valid is None:
            return super().forward(slots, inputs)
        assert self.input_norm is not None and self.gru is not None
        assert self.dense_k is not None and self.dense_v is not None
        assert self.dense_q is not None and self.q_norm is not None
        assert self.attention is not None

        mask = valid.to(inputs.dtype)  # (..., n_inputs)
        inputs_n = self.input_norm(inputs) * mask.unsqueeze(-1)
        k = self.dense_k(inputs_n)
        v = self.dense_v(inputs_n)
        for _ in range(self.num_iterations):
            q = self.dense_q(self.q_norm(slots))
            scale = float(q.shape[-1]) ** 0.5
            logits = torch.einsum("...qhd,...khd->...hqk", q / scale, k)
            neg_inf = torch.finfo(logits.dtype).min
            key_valid = mask.bool().reshape(*mask.shape[:-1], 1, 1, mask.shape[-1])
            logits = logits.masked_fill(~key_valid, neg_inf)
            attn = torch.softmax(logits, dim=-2)  # over the slot axis
            # Renormalise by the valid input mass (weighted mean over valid inputs only).
            mass = attn.sum(dim=-1, keepdim=True)
            attn = attn / (mass + self.epsilon)
            updates = torch.einsum("...hqk,...khd->...qhd", attn, v)
            updates = updates.squeeze(-2)
            slots = self.gru(slots, updates)
            if self.mlp is not None:
                slots = self.mlp(slots)
        return slots


@dataclass
class CorrectorPredictorTuple:
    """Upstream ``video.CorrectorPredictorTuple``."""

    corrected: torch.Tensor
    predicted: torch.Tensor


class Processor(nn.Module):
    """``savi.modules.video.Processor``: scan of corrector-then-predictor over time.

    The carried state is the *predicted* state (upstream ``new_state = predicted_state``), so the
    next step corrects the prediction of the previous step. ``corrected`` and ``predicted`` are
    both returned, which is what the SAVi decoder consumes.
    """

    def __init__(self, corrector: nn.Module, predictor: nn.Module):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor

    def forward(
        self,
        state: torch.Tensor,
        inputs: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, CorrectorPredictorTuple]:
        """``inputs`` has a time axis at position 1, as in the upstream ``nn.scan`` ``in_axes``."""
        n_frames = inputs.shape[1] if inputs is not None else 0
        corrected_frames, predicted_frames = [], []
        for t in range(n_frames):
            frame = inputs[:, t]
            step_mask = None if padding_mask is None else padding_mask[:, t]
            corrected = (
                self.corrector(state, frame, padding_mask=step_mask)
                if inputs is not None
                else state
            )
            predicted = self.predictor(corrected)
            corrected_frames.append(corrected)
            predicted_frames.append(predicted)
            state = predicted
        corrected_stack = torch.stack(corrected_frames, dim=1)
        predicted_stack = torch.stack(predicted_frames, dim=1)
        return state, CorrectorPredictorTuple(corrected=corrected_stack, predicted=predicted_stack)

    def step(
        self, state: torch.Tensor, new_inputs: torch.Tensor | None
    ) -> tuple[torch.Tensor, CorrectorPredictorTuple]:
        """One online step: correct with the new measurement, then predict the next state."""
        corrected = state if new_inputs is None else self.corrector(state, new_inputs)
        predicted = self.predictor(corrected)
        return predicted, CorrectorPredictorTuple(corrected=corrected, predicted=predicted)


def slot_attention_param_pairs(
    module: SlotAttention, prefix: str
) -> list[tuple[str, torch.Tensor, bool]]:
    """Map flattened Flax parameter paths to torch parameters of the port.

    Returns ``(flax_path, torch_parameter, transpose)`` triples; ``transpose=True`` means the
    Flax kernel is ``(in, out)`` and the torch parameter is ``(out, in)``.  Used by the parity
    suite to compare per-parameter gradients, so it must cover every Flax array of the module.
    """
    pairs: list[tuple[str, torch.Tensor, bool]] = []
    assert module.input_norm and module.q_norm and module.dense_q and module.dense_k
    assert module.dense_v and module.attention and module.gru
    # Flax construction order: the query norm is LayerNorm_0, the input norm is LayerNorm_1.
    for name, layer in (("LayerNorm_1", module.input_norm), ("LayerNorm_0", module.q_norm)):
        pairs.append((f"{prefix}/{name}/scale", layer.scale, False))
        pairs.append((f"{prefix}/{name}/bias", layer.bias, False))
    for name, dense in (
        ("general_dense_q_0", module.dense_q),
        ("general_dense_k_0", module.dense_k),
        ("general_dense_v_0", module.dense_v),
    ):
        pairs.append((f"{prefix}/{name}/kernel", dense.weight, True))
    gru_prefix = f"{prefix}/GRU_0/GRUCell_0"
    for name, layer in (
        ("ir", module.gru.ir),
        ("hr", module.gru.hr),
        ("iz", module.gru.iz),
        ("hz", module.gru.hz),
        ("in", module.gru.in_),
        ("hn", module.gru.hn),
    ):
        pairs.append((f"{gru_prefix}/{name}/kernel", layer.weight, True))
        if layer.bias is not None:
            pairs.append((f"{gru_prefix}/{name}/bias", layer.bias, False))
    if module.attention.multi_head and module.attention.out_proj is not None:
        pairs.append(
            (
                f"{prefix}/InvertedDotProductAttention_0/DenseGeneral_0/kernel",
                module.attention.out_proj.weight,
                True,
            )
        )
    if module.mlp is not None:
        mlp_prefix = f"{prefix}/MLP_0"
        if module.mlp.layernorm is not None:
            pairs.append((f"{mlp_prefix}/LayerNorm_0/scale", module.mlp.layernorm.scale, False))
            pairs.append((f"{mlp_prefix}/LayerNorm_0/bias", module.mlp.layernorm.bias, False))
        for i, layer in enumerate(module.mlp.hidden_layers):
            pairs.append((f"{mlp_prefix}/dense_mlp_{i}/kernel", layer.weight, True))
            pairs.append((f"{mlp_prefix}/dense_mlp_{i}/bias", layer.bias, False))
        assert module.mlp.output_layer is not None
        n = module.mlp.num_hidden_layers
        pairs.append((f"{mlp_prefix}/dense_mlp_{n}/kernel", module.mlp.output_layer.weight, True))
        pairs.append((f"{mlp_prefix}/dense_mlp_{n}/bias", module.mlp.output_layer.bias, False))
    return pairs


__all__ = [
    "REFERENCE_FLAX_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "CorrectorPredictorTuple",
    "FlaxDenseGeneral",
    "FlaxLayerNorm",
    "GRUCell",
    "InvertedDotProductAttention",
    "MLP",
    "Processor",
    "SlotAttention",
    "SlotAttentionMasked",
    "flatten_flax_params",
    "slot_attention_param_pairs",
]


class TransformerPredictor(nn.Module):
    """SAVi attention.Transformer self-attention path at the pinned R02 source.

    Post-norm, bias-bearing DenseGeneral Q/K/V and output projection; no dropout.
    Cross-attention is outside this predictor's scope. Parameters are checked
    against the original Flax module before the 512-wide profile is used.
    """

    def __init__(self, width=512, num_heads=4, mlp_size=2048, num_layers=4):
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.num_heads = num_heads
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    dict(
                        query=nn.Linear(width, width),
                        key=nn.Linear(width, width),
                        value=nn.Linear(width, width),
                        out=nn.Linear(width, width),
                        norm1=FlaxLayerNorm(width),
                        norm2=FlaxLayerNorm(width),
                        mlp=MLP(mlp_size, width).build(width),
                    )
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, slots):
        x = slots
        for block in self.blocks:
            b, n, d = x.shape
            q, k, v = [
                block[name](x).reshape(b, n, self.num_heads, d // self.num_heads)
                for name in ("query", "key", "value")
            ]
            attention = torch.einsum(
                "bqhd,bkhd->bhqk", q / (d // self.num_heads) ** 0.5, k
            ).softmax(-1)
            update = torch.einsum("bhqk,bkhd->bqhd", attention, v).reshape(b, n, d)
            x = block["norm1"](x + block["out"](update))
            x = block["norm2"](x + block["mlp"](x))
        return x

    def load_flax(self, flat, prefix="params"):
        for index, block in enumerate(self.blocks):
            base = f"{prefix}/TransformerBlock{index}"
            attn = f"{base}/MultiHeadDotProductAttention_0"
            for name in ("query", "key", "value", "out"):
                layer = block[name]
                kernel = _tensor(flat[f"{attn}/{name}/kernel"])
                kernel = kernel.reshape(layer.in_features, layer.out_features)
                with torch.no_grad():
                    layer.weight.copy_(kernel.t())
                    layer.bias.copy_(_tensor(flat[f"{attn}/{name}/bias"]).reshape(-1))
            block["norm1"].load_flax(flat, f"{base}/LayerNorm_0")
            block["norm2"].load_flax(flat, f"{base}/LayerNorm_1")
            block["mlp"].load_flax(flat, f"{base}/MLP_0")

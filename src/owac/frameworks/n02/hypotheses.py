"""Hypothesis initialisation, competition, proposal and the update cell.

How the K candidates stay distinct: they start from K different learned
embeddings combined with the *same* current measurement evidence, and they are
only ever merged through measurement-constrained weights. Latent Euclidean
distance between hypotheses is **not** treated as evidence that K physically
different explanations exist; the synthetic task and the readout probes in
``owac.evaluation.n02`` carry that burden.

The update cell is deliberately a small, readable ConvGRU (RAFT ``core/update.py``
is a compact public reference for the construction). Borrowing the cell is not
the contribution: the measurement energy, the contradiction map and the
dependency-propagated local reset are.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialShift(torch.autograd.Function):
    """Differentiable integer pixel shift with zero fill at the border."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        ctx.dy = dy
        ctx.dx = dx
        height, width = tensor.shape[-2:]
        out = tensor.new_zeros(tensor.shape)
        src_y0, src_y1 = max(0, -dy), min(height, height - dy)
        src_x0, src_x1 = max(0, -dx), min(width, width - dx)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[..., src_y0 + dy : src_y1 + dy, src_x0 + dx : src_x1 + dx] = tensor[
                ..., src_y0:src_y1, src_x0:src_x1
            ]
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = SpatialShift.apply(grad_output, -ctx.dy, -ctx.dx)
        return grad_input, None, None


def shift(tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    return SpatialShift.apply(tensor, dy, dx)


NEIGHBOURHOOD: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbour_stack(value: torch.Tensor) -> torch.Tensor:
    """Stack the four 4-neighbourhood shifts of ``[..., h, w]`` as ``[..., h, w, 4]``."""
    return torch.stack([shift(value, dy, dx) for dy, dx in NEIGHBOURHOOD], dim=-1)


def neighbour_validity(height: int, width: int, device: torch.device) -> torch.Tensor:
    """``[4, h, w]`` float mask, 0.0 where a neighbour lies outside the grid."""
    rows = torch.arange(height, device=device).view(height, 1).expand(height, width)
    cols = torch.arange(width, device=device).view(1, width).expand(height, width)
    masks = []
    for dy, dx in NEIGHBOURHOOD:
        valid_y = (rows + dy).clamp(0, height - 1).eq(rows + dy)
        valid_x = (cols + dx).clamp(0, width - 1).eq(cols + dx)
        masks.append((valid_y & valid_x).to(torch.float32))
    return torch.stack(masks, dim=0)


def coordinate_features(height: int, width: int, dim: int, device: torch.device) -> torch.Tensor:
    """Fixed 2-D sinusoidal position code ``[dim, h, w]`` for the readout and cell."""
    assert dim >= 4, "coordinate features need at least 4 channels"
    half = dim // 2
    quarter = max(1, half // 2)
    rows = (
        torch.arange(height, device=device).to(torch.float32).view(height, 1).expand(height, width)
    )
    cols = torch.arange(width, device=device).to(torch.float32).view(1, width).expand(height, width)
    features = []
    for index in range(quarter):
        frequency = float(2**index)
        features.append(torch.sin(rows / height * 3.141592653589793 * frequency))
        features.append(torch.cos(rows / height * 3.141592653589793 * frequency))
        features.append(torch.sin(cols / width * 3.141592653589793 * frequency))
        features.append(torch.cos(cols / width * 3.141592653589793 * frequency))
    stacked = torch.stack(features, dim=0)
    if stacked.shape[0] < dim:
        pad = stacked.new_zeros(dim - stacked.shape[0], height, width)
        stacked = torch.cat([stacked, pad], dim=0)
    return stacked[:dim]


def _total_channels(parts: list[torch.Tensor]) -> int:
    return sum(part.shape[2] for part in parts)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class HypothesisInitializer(nn.Module):
    """Forms K candidate explanations from the measurement basis and a causal code.

    ``causal_context`` may be ``None``; a missing history must stay missing
    rather than being replaced by a zero vector that looks like real context.
    """

    def __init__(
        self,
        width: int,
        K: int,
        *,
        context_dim: int = 0,
        code_scale: float = 0.5,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.K = K
        self.context_dim = context_dim
        hidden = hidden or max(16, width)
        self.basis = nn.Conv2d(width, hidden, 1)
        self.code = nn.Embedding(K, hidden)
        self.mix = nn.Sequential(
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, width, 1),
        )
        self.context = nn.Linear(context_dim, K * hidden) if context_dim else None
        self.code_scale = float(code_scale)
        nn.init.normal_(self.code.weight, std=0.5)

    def forward(
        self,
        basis: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``basis`` ``[B*V, width, h, w]`` -> ``[B*V, K, width, h, w]``."""
        shared = self.basis(basis)  # [N, hidden, h, w]
        codes = self.code.weight  # [K, hidden]
        offset = codes.view(1, self.K, -1, 1, 1)
        if self.context is not None:
            if context is None:
                context = basis.new_zeros(basis.shape[0], self.context_dim)
            bias = self.context(context).view(basis.shape[0], self.K, -1, 1, 1)
            offset = offset + bias
        combined = shared.unsqueeze(1) + self.code_scale * offset
        combined = combined.reshape(basis.shape[0] * self.K, -1, basis.shape[-2], basis.shape[-1])
        mixed = self.mix(combined)
        return mixed.reshape(basis.shape[0], self.K, self.width, basis.shape[-2], basis.shape[-1])


def flatten_hypotheses(value: torch.Tensor, name: str) -> torch.Tensor:
    """``[B,V,K,C,h,w]`` -> ``[B*V*K,C,h,w]``.

    The ``C`` candidates are separate batch elements of the update cell, so a
    candidate can never mix its channels with another candidate's.
    """
    if value.dim() != 6:
        raise ValueError(f"{name} must be [B,V,K,C,h,w], got {tuple(value.shape)}")
    b, v, k = value.shape[:3]
    return value.reshape(b * v * k, *value.shape[3:])


def evidence_map(value: torch.Tensor, name: str, K: int, grid: tuple[int, int]) -> torch.Tensor:
    """Normalise an evidence map to ``[B*V*K,1,h,w]`` for the proposal head.

    Three layouts are accepted: ``[B,V,K,h,w]`` (one map per candidate),
    ``[B,V,1,h,w]`` (one map shared by every candidate) and ``[B,V,C,h,w]`` with
    ``C`` neither 1 nor K, which is a shared *feature* map and is reduced to its
    channel mean. The hypothesis axis is never silently broadcast from a feature
    map, so a wrong tensor cannot be mistaken for per-candidate evidence.
    """
    if value.dim() == 4:
        value = value.unsqueeze(2)
    if value.dim() != 5:
        raise ValueError(f"{name} must be [B,V,K,h,w] or [B,V,C,h,w], got {tuple(value.shape)}")
    b, v, k = value.shape[:3]
    if tuple(value.shape[-2:]) != tuple(grid):
        raise ValueError(
            f"{name} spatial {tuple(value.shape[-2:])} does not match the field grid {tuple(grid)}"
        )
    if k not in {1, K}:
        value = value.mean(dim=2, keepdim=True)
        k = 1
    flat = value.reshape(b * v, k, 1, *grid)
    if k == 1:
        flat = flat.expand(-1, K, -1, -1, -1)
    return flat.reshape(b * v * K, 1, *grid)


class HypothesisUpdateCell(nn.Module):
    """Shared-weight ConvGRU cell conditioned on measurement residuals.

    At a location ``u`` the cell sees, as separate channel groups: the current
    cell state, the neighbourhood message over the recorded dependency edges, the
    measurement residual, the prior context and the fixed position code. Nothing
    is pre-blended, so the prior can never leak into the measurement term.

    Inputs use the ``[B,V,K,...]`` convention and are flattened to
    ``[B*V*K,...]`` internally. The first convolution is lazy, so the channel
    bookkeeping follows the actual inputs instead of a hand-maintained formula;
    :meth:`configuration` reports the resolved widths for the cost record.
    """

    def __init__(
        self,
        width: int,
        *,
        kernel: int = 3,
        kind: str = "convgru",
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if kind not in {"convgru", "residual"}:
            raise ValueError(f"unknown cell kind: {kind!r}")
        self.kind = kind
        self.width = int(width)
        self.context_dim = int(context_dim)
        padding = kernel // 2
        self.input = nn.LazyConv2d(self.width * 2, kernel, padding=padding)
        self.norm = nn.GroupNorm(_group_count(self.width * 2), self.width * 2)
        self.hidden = nn.LazyConv2d(self.width, kernel, padding=padding)
        self.hidden_norm = nn.GroupNorm(_group_count(self.width), self.width)
        self.resolved_input_channels: int | None = None
        self.resolved_hidden_channels: int | None = None

    def configuration(self) -> dict[str, object]:
        return {
            "width": self.width,
            "kind": self.kind,
            "input_channels": self.resolved_input_channels,
            "hidden_input_channels": self.resolved_hidden_channels,
            "context_dim": self.context_dim,
        }

    @staticmethod
    def _expand(value: torch.Tensor, K: int, name: str) -> torch.Tensor:
        """``[B,V,K,h,w]``/``[B,V,K,C,h,w]``/``[B,V,1,C,h,w]`` -> ``[B*V*K,C,h,w]``.

        A map without a channel axis is read as a single-channel evidence map for
        this candidate; the hypothesis axis is never silently broadcast, so a
        wrong tensor cannot be mistaken for per-candidate evidence.
        """
        if value.dim() == 5 and value.shape[2] in {1, K}:
            # [B,V,K,h,w] is a one-channel per-candidate map.
            value = value.unsqueeze(3)
        elif value.dim() == 5:
            # [B,V,C,h,w] with C != K is a shared feature map: add the K axis.
            value = value.unsqueeze(2)
        if value.dim() != 6:
            raise ValueError(
                f"{name} must be [B,V,K,h,w], [B,V,C,h,w] or [B,V,K,C,h,w], "
                f"got {tuple(value.shape)}"
            )
        if value.shape[2] == 1 and K > 1:
            value = value.expand(-1, -1, K, -1, -1, -1)
        elif value.shape[2] != K:
            raise ValueError(
                f"{name} carried {value.shape[2]} hypothesis groups, expected 1 or K={K}"
            )
        return flatten_hypotheses(value, name)

    def forward(
        self,
        state: torch.Tensor,
        message: torch.Tensor,
        residual: torch.Tensor,
        prior: torch.Tensor,
        position: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``state`` ``[B,V,K,W,h,w]``; returns the candidate update, same shape."""
        if state.dim() != 6:
            raise ValueError(f"state must be [B,V,K,W,h,w], got {tuple(state.shape)}")
        batch, views, K, width = state.shape[:4]
        spatial = tuple(state.shape[4:])
        state_flat = flatten_hypotheses(state, "state")
        terms = [
            state_flat,
            self._expand(message, K, "message"),
            self._expand(residual, K, "residual"),
            self._expand(prior, K, "prior"),
            self._expand(position, K, "position"),
        ]
        if self.context_dim:
            if context is None:
                context = state.new_zeros(batch, views, K, self.context_dim, *spatial)
            terms.append(self._expand(context, K, "context"))
        combined = torch.cat(terms, dim=1)
        gates = self.norm(self.input(combined))
        reset, update = gates.chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        # The reset gate only gates the recurrent term, not the evidence terms.
        hidden_input = torch.cat([state_flat * reset, *terms[1:]], dim=1)
        self.resolved_input_channels = int(combined.shape[1])
        self.resolved_hidden_channels = int(hidden_input.shape[1])
        candidate = torch.tanh(self.hidden_norm(self.hidden(hidden_input)))
        if self.kind == "residual":
            update = torch.ones_like(update)
        updated = (1.0 - update) * state_flat + update * candidate
        if updated.shape[0] != batch * views * K:
            raise ValueError(
                f"cell produced {updated.shape[0]} grouped rows, expected {batch * views * K}"
            )
        return updated.reshape(batch, views, K, width, *spatial)


class ProposalHead(nn.Module):
    """Proposes a fresh candidate explanation from the local measurement evidence.

    Used after a contradiction: the reset region is re-initialised from the
    current measurement residual plus the remaining context instead of from the
    explanation that was just refuted.
    """

    def __init__(self, width: int, K: int, *, kernel: int = 3) -> None:
        super().__init__()
        self.width = int(width)
        self.K = int(K)
        padding = kernel // 2
        self.proj = nn.Sequential(
            nn.Conv2d(self.width + 3, self.width, kernel, padding=padding),
            nn.GroupNorm(_group_count(self.width), self.width),
            nn.SiLU(),
            nn.Conv2d(self.width, self.width, kernel, padding=padding),
        )

    def forward(
        self,
        state: torch.Tensor,
        residual: torch.Tensor,
        contradiction: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """``state`` ``[B,V,K,W,h,w]``; returns a proposed field of the same shape."""
        if state.dim() != 6:
            raise ValueError(f"state must be [B,V,K,W,h,w], got {tuple(state.shape)}")
        batch, views, K, width = state.shape[:4]
        spatial = tuple(state.shape[4:])
        flat = flatten_hypotheses(state, "state")
        evidence = [
            evidence_map(residual, "residual", K, spatial),
            evidence_map(contradiction, "contradiction", K, spatial),
            evidence_map(position, "position", K, spatial),
        ]
        proposed = self.proj(torch.cat([flat, *evidence], dim=1))
        return proposed.reshape(batch, views, K, width, *spatial)


def diversity_penalty(h: torch.Tensor, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean pairwise normalised distance and a collapse proxy for ``[B,V,K,D,h,w]``.

    Returns ``(mean_distance, collapsed_fraction)``. The distance is between
    mean-pooled hypothesis states, normalised by their own scale so it is
    comparable across widths. ``collapsed_fraction`` is the share of hypothesis
    pairs below ``eps`` relative distance: a diagnostic that the K candidates
    degenerated into one explanation. This is a *diagnostic*, not evidence that K
    physically distinct explanations exist.
    """
    batch, views, K, width, height, width_px = h.shape
    pooled = h.mean(dim=(-1, -2))  # [B, V, K, D]
    norm = pooled.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    unit = pooled / norm
    similarity = torch.einsum("bvkd,bvld->bvkl", unit, unit)
    eye = torch.eye(K, device=h.device, dtype=torch.bool).view(1, 1, K, K)
    pair = similarity.masked_select(~eye)
    if pair.numel() == 0:
        zero = h.new_zeros(())
        return zero, zero
    distance = (1.0 - pair).clamp_min(0.0)
    collapsed = (distance < eps).to(h.dtype).mean()
    return distance.mean(), collapsed


def geometry_spread(geometry: torch.Tensor) -> torch.Tensor:
    """Mean pairwise distance of the declared latent geometry ``[B,V,K,G]``."""
    batch, views, K, dim = geometry.shape
    flat = geometry.reshape(batch * views, K, dim)
    diff = flat[:, :, None, :] - flat[:, None, :, :]
    distance = diff.norm(dim=-1)
    eye = torch.eye(K, device=geometry.device, dtype=torch.bool).view(1, K, K)
    pair = distance.masked_select(~eye)
    if pair.numel() == 0:
        return geometry.new_zeros(())
    return pair.mean()

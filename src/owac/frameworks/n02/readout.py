"""Backend readouts: MAP selection and grouped hypothesis tokens.

Two readouts are required because they answer different questions.

* :func:`map_hypothesis` picks one explanation (globally best, or the best
  consistent local combination) and keeps the selection index plus uncertainty
  diagnostics. It is the honest, most direct baseline.
* :func:`hypothesis_tokens` gives every candidate its own token group carrying a
  hypothesis id, position, support and weight, so a backend can in principle
  keep the ambiguity instead of averaging it away.

Neither function may average incompatible candidates into a middle geometry.
That failure mode is measured by the probes in :func:`two_candidate_probe`,
which compares a direct mean readout against MAP and grouped readouts on a task
where the two candidates imply different actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.core.encoder_contract import EncoderOutput, VisualTokens
from owac.frameworks.n02.hypotheses import coordinate_features


@dataclass(slots=True)
class StructuredRepresentation:
    """What an N02 step returns as its ``representation``.

    ``h``/``weights`` are the hypothesis field and its measurement-constrained
    weights. ``geometry`` is the declared latent geometry each candidate
    predicts, which is what the mechanism losses supervise; it is never an
    intermediate image.
    """

    h: torch.Tensor  # [B, V, K, D, h, w]
    weights: torch.Tensor  # [B, V, K]
    measurement_energy: torch.Tensor  # [B, V, K]
    prior_energy: torch.Tensor  # [B, V, K]
    relation_energy: torch.Tensor  # [B, V, K]
    geometry: torch.Tensor  # [B, V, K, geometry_dim]
    valid: torch.Tensor | None = None  # [B, V] camera validity
    support: torch.Tensor | None = None  # [B, V, K, h, w]
    step_index: int = 0
    diagnostics: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.h.shape[2]

    @property
    def V(self) -> int:
        return self.h.shape[1]

    def map_indices(self) -> torch.Tensor:
        """``[B, V]`` index of the highest-weighted explanation per view."""
        return self.weights.argmax(dim=-1)

    def map_geometry(self) -> torch.Tensor:
        """``[B, V, geometry_dim]`` geometry of the selected explanation."""
        index = self.map_indices().unsqueeze(-1).unsqueeze(-1)
        return self.geometry.gather(2, index.expand(-1, -1, 1, self.geometry.shape[-1])).squeeze(2)

    def top_weight(self) -> torch.Tensor:
        return self.weights.max(dim=-1).values

    def entropy(self) -> torch.Tensor:
        safe = self.weights.clamp_min(1.0e-12)
        return -(safe * safe.log()).sum(dim=-1)

    def unresolved_ambiguity(self, threshold: float = 0.25) -> torch.Tensor:
        """``[B, V]`` bool: no single explanation is decisively supported.

        A uniform posterior over K has top weight ``1/K``; the threshold states
        how far above uniform the leading candidate must be before the field is
        called resolved. This is a declared diagnostic, not a calibrated
        probability statement.
        """
        uniform = 1.0 / max(self.K, 1)
        return self.top_weight() <= uniform + threshold * (1.0 - uniform)


@dataclass(slots=True)
class ReadoutOutput:
    """Tokens plus the diagnostics a caller needs to audit the readout."""

    tokens: VisualTokens
    token_weight_sum: torch.Tensor  # [B, V, K] sum of weights carried by each group
    hypothesis_index: torch.Tensor  # [B, V, L] group id per token
    mean_collapse: torch.Tensor | None = None  # geometry of the weighted mean
    map_geometry: torch.Tensor | None = None
    diagnostics: dict = field(default_factory=dict)


class TokenReadout(nn.Module):
    """Turns a hypothesis field into π0.5-sized visual tokens.

    The per-camera token budget is fixed (default 256 tokens of 2048 dims). With
    ``K`` candidates the budget is split into ``K`` groups of ``L = budget / K``
    tokens, so adding candidates never silently buys more backend input.
    """

    def __init__(
        self,
        width: int,
        K: int,
        *,
        token_dim: int = 2048,
        hidden: int | None = None,
        position_channels: int = 4,
        num_views_max: int = 8,
    ) -> None:
        super().__init__()
        self.width = width
        self.K = K
        self.token_dim = token_dim
        self.position_channels = position_channels
        self.num_views_max = num_views_max
        hidden = hidden or max(64, min(token_dim, width * 4))
        self.context_channels = 2  # candidate weight + runner-up weight
        self.project = nn.Sequential(
            nn.Linear(width + position_channels + self.context_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, token_dim),
        )
        self.camera_embedding = nn.Embedding(num_views_max, width)
        self.hypothesis_embedding = nn.Embedding(K, width)

    def forward(
        self,
        representation: StructuredRepresentation,
        *,
        token_budget: int,
        mode: str = "tokens",
        camera_ids: tuple = (),
        positions: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validity: torch.Tensor | None = None,
    ) -> ReadoutOutput:
        h = representation.h
        batch, views, K, width, height, width_px = h.shape
        if mode not in {"map", "tokens"}:
            raise ValueError(f"unknown readout mode: {mode!r}")
        groups = 1 if mode == "map" else K
        if token_budget % groups != 0:
            raise ValueError(
                f"token_budget={token_budget} must be divisible by the group count {groups}"
            )
        tokens_per_group = token_budget // groups
        grid = self._pool_grid(height, width_px, tokens_per_group)
        keep_h, keep_w = grid
        if keep_h * keep_w < tokens_per_group:
            raise ValueError(
                f"pooling grid {grid} cannot supply {tokens_per_group} tokens per group"
            )
        # The budget is a contract: when the grid has surplus cells, the extra ones
        # are dropped deterministically (row-major order) rather than emitting more
        # tokens than the caller asked for.
        keep_count = keep_h * keep_w
        pooled = F.adaptive_avg_pool2d(
            h.reshape(batch * views * K, width, height, width_px), (keep_h, keep_w)
        ).reshape(batch, views, K, width, keep_h, keep_w)

        if mode == "map":
            index = representation.map_indices()  # [B, V]
            gather_index = index.view(batch, views, 1, 1, 1, 1).expand(
                -1, -1, 1, width, keep_h, keep_w
            )
            pooled = pooled.gather(2, gather_index)  # [B, V, 1, W, h', w']
            weights = representation.weights.gather(2, index.unsqueeze(-1))  # [B, V, 1]
            group_ids = index
        else:
            weights = representation.weights
            group_ids = torch.arange(K, device=h.device).view(1, 1, K).expand(batch, views, K)

        position = self._position_features(keep_h, keep_w, h)
        camera = self._camera_features(views, keep_h, keep_w, h, camera_ids)
        hypothesis_code = self.hypothesis_embedding(
            torch.arange(K if mode == "tokens" else 1, device=h.device).clamp_max(self.K - 1)
        )
        if mode == "map":
            hypothesis_code = hypothesis_code.view(1, 1, 1, -1, 1, 1)
        else:
            hypothesis_code = hypothesis_code.view(1, 1, K, -1, 1, 1)
        # Two explicit reliability channels per emitted token group: the group's
        # own weight and the best competing weight. Both are real consumable
        # fields carried into the backend, not logging.
        weights = representation.weights  # [B,V,K], index order of the candidates
        sorted_weights = weights.sort(dim=-1, descending=True).values
        best_scalar = sorted_weights[..., :1]  # [B,V,1]
        runner_scalar = (
            sorted_weights[..., 1:2]
            if sorted_weights.shape[-1] > 1
            else torch.zeros_like(best_scalar)
        )
        if mode == "map":
            selected = index.view(batch, views, 1)  # [B,V,1]
            best_group = weights.gather(2, selected)
            # The competing weight for the selected candidate: the global runner-up
            # when the best candidate was selected, otherwise the global best.
            runner_group = torch.where(
                selected == index.view(batch, views, 1), runner_scalar, best_scalar
            )
            runner_group = torch.where(
                weights.gather(2, selected) >= best_scalar, runner_scalar, best_scalar
            )
        else:
            best_group = weights
            top_two = weights.topk(min(2, K), dim=-1).values
            runner_group = (
                top_two[..., 1:2].expand(-1, -1, K)
                if top_two.shape[-1] > 1
                else torch.zeros_like(weights)
            )
        reliability = torch.cat([best_group, runner_group], dim=-1)  # [B,V,G,2]
        weight_feature = reliability.view(batch, views, -1, 2, 1, 1).expand(
            -1, -1, -1, -1, keep_h, keep_w
        )
        # The camera and hypothesis identities are added into the feature width
        # (they are offsets, not extra channels), so the projection input width is
        # exactly width + position_channels + 2 and cannot drift silently.
        identity = camera + hypothesis_code
        features = torch.cat(
            [
                pooled + identity,
                position.expand(batch, views, pooled.shape[2], -1, keep_h, keep_w),
                weight_feature,
            ],
            dim=3,
        )
        token_values = self.project(features.movedim(3, -1))  # [B,V,G,h',w',token_dim]
        token_values = token_values.reshape(batch, views, -1, self.token_dim)
        if token_values.shape[2] != groups * tokens_per_group:
            # Flattened cell order is (group, row, col); keeping the first
            # ``tokens_per_group`` cells of each group preserves spatial overlap.
            reshaped = token_values.reshape(batch, views, groups, keep_count, self.token_dim)
            token_values = reshaped[:, :, :, :tokens_per_group, :].reshape(
                batch, views, groups * tokens_per_group, self.token_dim
            )

        token_valid = self._token_validity(
            representation,
            views,
            keep_h,
            keep_w,
            groups=groups,
            validity=validity,
            camera_ids=camera_ids,
            keep_count=tokens_per_group,
        )
        if token_valid is not None:
            token_values = token_values * token_valid.unsqueeze(-1)
        values = token_values.reshape(batch, views * token_values.shape[2], self.token_dim)
        mask = None
        if token_valid is not None:
            mask = token_valid.reshape(batch, -1)
        if positions is None:
            token_positions = self._grid_positions(keep_h, keep_w, views, h)
        else:
            token_positions = positions
        camera_ids_out = camera_ids or tuple(f"camera_{index}" for index in range(views))
        collapse = None
        if mode == "tokens":
            collapse = torch.einsum("bvk,bvkg->bvg", weights, representation.geometry)
        return ReadoutOutput(
            tokens=VisualTokens(
                values=values,
                mask=mask,
                camera_ids=camera_ids_out,
                positions=token_positions,
                scale=None if scale is None else scale,
                validity=token_valid,
                uncertainty=representation.weights,
            ),
            token_weight_sum=weights,
            hypothesis_index=self._group_index(
                group_ids, tokens_per_group=tokens_per_group, views=views
            ),
            mean_collapse=collapse,
            map_geometry=representation.map_geometry(),
            diagnostics={
                "mode": mode,
                "tokens_per_group": tokens_per_group,
                "grid": [keep_h, keep_w],
                "groups": groups,
            },
        )

    @staticmethod
    def _group_index(group_ids: torch.Tensor, *, tokens_per_group: int, views: int) -> torch.Tensor:
        """``[B,V,L]`` group id per emitted token; L = views * tokens_per_group."""
        batch = group_ids.shape[0]
        per_view = group_ids.reshape(batch, views, -1).repeat_interleave(tokens_per_group, dim=-1)
        return per_view.reshape(batch, -1)

    @staticmethod
    def _pool_grid(height: int, width: int, tokens: int) -> tuple[int, int]:
        """Choose a grid whose cell count is >= the token count, aspect preserving."""
        if tokens <= 0:
            raise ValueError("token budget must be positive")
        best = (1, 1)
        best_score = None
        for rows in range(1, tokens + 1):
            cols = -(-tokens // rows)
            if rows * cols < tokens:
                continue
            aspect = abs((rows / cols) - (height / max(width, 1)))
            score = (aspect, rows * cols - tokens)
            if best_score is None or score < best_score:
                best_score = score
                best = (rows, cols)
        return best

    def _position_features(self, rows: int, cols: int, like: torch.Tensor) -> torch.Tensor:
        features = coordinate_features(rows, cols, self.position_channels, like.device)
        return features.to(like.dtype).view(1, 1, 1, self.position_channels, rows, cols)

    def _camera_features(
        self, views: int, rows: int, cols: int, like: torch.Tensor, camera_ids: tuple
    ) -> torch.Tensor:
        del camera_ids
        index = torch.arange(views, device=like.device).clamp_max(self.num_views_max - 1)
        embedding = self.camera_embedding(index)  # [V, W]
        return (
            embedding.view(1, views, 1, self.width, 1, 1)
            .expand(-1, -1, 1, -1, rows, cols)
            .to(like.dtype)
        )

    def _token_validity(
        self,
        representation: StructuredRepresentation,
        views: int,
        rows: int,
        cols: int,
        *,
        groups: int,
        validity: torch.Tensor | None,
        camera_ids: tuple,
        keep_count: int | None = None,
    ) -> torch.Tensor | None:
        batch = representation.h.shape[0]
        camera = representation.valid
        if camera is None and validity is not None:
            camera = validity
        if camera is None and camera_ids:
            camera = torch.ones(batch, views, device=representation.h.device, dtype=torch.bool)
        if camera is None:
            return None
        camera = camera.to(torch.bool)
        if camera.shape[1] != views:
            camera = camera[:, :views]
        support = representation.support
        if support is None:
            spatial = torch.ones(
                batch, views, 1, rows, cols, device=representation.h.device, dtype=torch.bool
            )
        else:
            reduced = F.adaptive_max_pool2d(
                support.to(torch.float32).reshape(-1, 1, *support.shape[-2:]), (rows, cols)
            )
            spatial = (reduced > 0).reshape(batch, views, support.shape[2], rows, cols)
            spatial = spatial.any(dim=2, keepdim=True)
        if groups == 1:
            spatial = spatial[:, :, :1]
        if spatial.shape[1] != groups:
            sample = spatial if spatial.shape[1] == 1 else spatial[:, :1]
            spatial = sample.expand(-1, groups, -1, -1, -1)
        combined = camera.view(batch, views, 1, 1, 1) & spatial
        combined = combined.expand(-1, -1, -1, rows, cols).reshape(
            batch, views, groups, rows * cols
        )
        if keep_count is not None:
            combined = combined[:, :, :, :keep_count]
        return combined.reshape(batch, views, -1)

    @staticmethod
    def _grid_positions(rows: int, cols: int, views: int, like: torch.Tensor) -> torch.Tensor:
        row_index = torch.arange(rows, device=like.device, dtype=like.dtype).view(rows, 1, 1)
        col_index = torch.arange(cols, device=like.device, dtype=like.dtype).view(1, cols, 1)
        grid = torch.stack(
            [row_index.expand(rows, cols, 1), col_index.expand(rows, cols, 1)], dim=-1
        ).reshape(rows * cols, 2)
        return grid.unsqueeze(1).expand(-1, views, -1)  # [L, V, 2]


def map_hypothesis(
    representation: StructuredRepresentation,
    token_budget: int,
    *,
    readout: TokenReadout,
    camera_ids: tuple = (),
    validity: torch.Tensor | None = None,
) -> ReadoutOutput:
    """Select the single best explanation and read it out as tokens."""
    return readout(
        representation,
        token_budget=token_budget,
        mode="map",
        camera_ids=camera_ids,
        validity=validity,
    )


def hypothesis_tokens(
    representation: StructuredRepresentation,
    token_budget: int,
    *,
    readout: TokenReadout,
    camera_ids: tuple = (),
    validity: torch.Tensor | None = None,
) -> ReadoutOutput:
    """Give every candidate its own token group within the same budget."""
    return readout(
        representation,
        token_budget=token_budget,
        mode="tokens",
        camera_ids=camera_ids,
        validity=validity,
    )


@dataclass(slots=True)
class ProbeResult:
    """Outcome of the two-incompatible-candidate readout probe."""

    action: torch.Tensor  # [B, action_dim] decoded action for the probe task
    geometry: torch.Tensor  # [B, geometry_dim] geometry the action was based on
    interpolation_fraction: torch.Tensor  # [B] fraction of the ambiguity collapsed away
    selected_index: torch.Tensor  # [B] MAP selection
    readout: str = ""


def two_candidate_probe(
    representation: StructuredRepresentation,
    *,
    action_fn: Any,
    mode: str = "map",
) -> ProbeResult:
    """Probe whether a readout keeps two incompatible candidates apart.

    ``action_fn`` maps declared geometry ``[B, G]`` to an action ``[B, A]``. The
    probe compares the readout's own geometry with the geometry of the
    action-relevant candidate, and reports how much of the difference to the
    discarded candidate was averaged away:

    ``interpolation = |g_readout - g_selected| / |g_other - g_selected|``

    A direct mean readout with equal weights gives ``0.5`` on two candidates;
    MAP gives ``0.0``. A backend that silently fuses grouped tokens degenerates
    towards the mean, which this number makes visible.
    """
    weights = representation.weights
    if representation.K < 2:
        raise ValueError("the two-candidate probe needs K >= 2")
    index = representation.weights.argmax(dim=-1)  # [B, V]
    geometry = representation.geometry
    selected = representation.map_geometry()
    if mode == "mean":
        mean = torch.einsum("bvk,bvkg->bvg", weights, geometry)
        used = mean
    elif mode in {"map", "tokens"}:
        used = selected
    else:
        raise ValueError(f"unknown probe mode: {mode!r}")
    # The competitor must be a *different* candidate. ``topk`` breaks an exact tie
    # by index order while ``argmax`` may return either, so the second index is
    # chosen as the best candidate whose index differs from the selection; otherwise
    # a perfect tie reported an interpolation of 1e8.
    order = weights.argsort(dim=-1, descending=True, stable=True)  # [B,V,K]
    selected_index = representation.map_indices()
    # Keep the weight order but push the selected candidate to the end, so the best
    # remaining candidate is the first entry.
    is_other = order != selected_index.unsqueeze(-1)
    other_index = torch.where(is_other, order, order.new_full((), weights.shape[-1])).amin(dim=-1)
    other = geometry.gather(
        2, other_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, geometry.shape[-1])
    ).squeeze(2)
    difference = (other - selected).norm(dim=-1)
    deviation = (used - selected).norm(dim=-1)
    # With two identical candidates the interpolation question is undefined; report
    # 0 rather than dividing by a clamped constant.
    degenerate = difference <= 1.0e-8
    interpolation = torch.where(
        degenerate, torch.zeros_like(deviation), deviation / difference.clamp_min(1.0e-8)
    )
    action = action_fn(used.reshape(-1, used.shape[-1])).reshape(used.shape[0], used.shape[1], -1)
    return ProbeResult(
        action=action,
        geometry=used,
        interpolation_fraction=interpolation,
        selected_index=index,
        readout=mode,
    )


def representation_from_output(output: EncoderOutput) -> StructuredRepresentation:
    """Recover the structured representation from a contract :class:`EncoderOutput`."""
    representation = output.representation
    if not isinstance(representation, StructuredRepresentation):
        raise TypeError(
            "encoder output does not carry a StructuredRepresentation; "
            f"got {type(representation).__name__}"
        )
    return representation

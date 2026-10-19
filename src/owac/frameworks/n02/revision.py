"""Contradiction localisation, dependency propagation and local recomposition.

The one rule that distinguishes N02 from ordinary recurrent inference lives
here: when a new measurement refutes a candidate explanation at a location, the
revision does not restart everything and does not re-normalise the posterior
until the answer changes. It

1. marks the locations and candidates the measurement contradicts,
2. propagates that priority **along the recorded dependency edges** for a bounded
   number of hops,
3. recomputes a shared cell update only inside the selected region,
4. downweights the refuted candidates and lets a *proposal* head rebuild the
   affected local structure from the current residual, and
5. recalculates the full energy on the same frozen measurement set.

``propagation_enabled``, ``reset_contradicted`` and ``proposal_enabled`` are real
switches: turning them off yields the ablations, and none of them works by
setting ``beta_prior`` to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn

from owac.frameworks.n02.config import N02Config
from owac.frameworks.n02.hypotheses import (
    HypothesisUpdateCell,
    ProposalHead,
    neighbour_stack,
)
from owac.frameworks.n02.state import HypothesisState, RevisionState


@dataclass(slots=True)
class RevisionOutput:
    """Result of one revision cycle."""

    h: torch.Tensor
    priority: torch.Tensor  # [B, V, h, w] scalar priority for diagnostics
    contradiction: torch.Tensor  # [B, V, K, h, w] pre-propagation
    propagated: torch.Tensor  # [B, V, K, h, w] post-propagation update selection
    reset: torch.Tensor  # [B, V, K, h, w]
    updated_fraction: float
    contradicted_fraction: float
    selected_candidates: torch.Tensor  # [B, V, K] bool
    iteration: int
    converged: bool
    revision: RevisionState | None = None  # new bookkeeping; the input state is untouched
    trace: dict = field(default_factory=dict)


def _validity_for(
    valid: torch.Tensor,
    batch: int,
    views: int,
    height: int,
    width: int,
    neighbours: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coerce a stored neighbour-validity mask to ``[B,V,h,w,4]``, dtype cast.

    The field grid follows the camera spec, so a stored mask can differ from the
    requested grid after a resize. Resizing keeps boundary semantics explicit
    instead of silently broadcasting a mask that no longer matches.
    """
    mask = valid.to(dtype)
    if mask.dim() == 4:  # [B,V,h,w] with no neighbour axis
        mask = mask.unsqueeze(-1)
    if mask.dim() != 5:
        raise ValueError(f"unexpected neighbour-validity shape {tuple(valid.shape)}")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, -1, -1, -1, -1)
    if mask.shape[1] == 1 and views > 1:
        mask = mask.expand(-1, views, -1, -1, -1)
    if mask.shape[-1] != neighbours:
        mask = (
            mask[..., :neighbours]
            if mask.shape[-1] > neighbours
            else mask.expand(-1, -1, -1, -1, neighbours)
        )
    source_height, source_width = mask.shape[2], mask.shape[3]
    if (source_height, source_width) != (height, width):
        flat = mask.permute(0, 1, 4, 2, 3).reshape(-1, 1, source_height, source_width)
        flat = torch.nn.functional.interpolate(flat.float(), size=(height, width), mode="nearest")
        mask = flat.reshape(batch, views, neighbours, height, width).permute(0, 1, 3, 4, 2)
    return mask.contiguous()


class RevisionController(nn.Module):
    """Owns the update cell, the proposal head and the per-cycle bookkeeping."""

    def __init__(self, config: N02Config, *, width: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.revision = config.revision
        self.width = int(width or config.D)
        # The cell sees explicit evidence groups, never one blended term:
        # per-hypothesis residual (K), the prior context (K) and the position code.
        self.position_channels = 4
        self.hypothesis_count = config.K
        self.cell = HypothesisUpdateCell(
            self.width,
            kernel=config.cell_kernel,
            kind=config.cell_kind,
            context_dim=self.width,
        )
        self.proposal = ProposalHead(self.width, config.K, kernel=config.cell_kernel)
        # Gate inputs: mean residual (1), prior level (1), position code (1).
        self.location_update = torch.nn.Conv2d(3, 1, 1)
        # Small random kernel + negative bias: the gate starts nearly closed (a
        # local recomposition rather than a full-field rewrite) but still receives
        # gradient. An exactly-zero kernel would make "learnable gate" false.
        torch.nn.init.normal_(self.location_update.weight, std=0.05)
        torch.nn.init.constant_(self.location_update.bias, -2.0)

    # ---------------------------------------------------------------- evidence
    def contradiction_map(
        self,
        normalized_residual: torch.Tensor,
        *,
        update_ema: bool = True,
        revision: RevisionState | None = None,
    ) -> torch.Tensor:
        """``[B,V,K,h,w]`` bool: this candidate is refuted at this location.

        The test is relative to the best explanation available here: a candidate
        must exceed ``max(relative_margin * best_fit, contradiction_threshold)``,
        and the running estimate of that boundary keeps a globally noisy frame
        from flagging every location at once. "No candidate explains this
        measurement" is a separate claim, reported by
        :meth:`all_candidates_bad`.
        """
        residual = normalized_residual
        threshold = self.revision.contradiction_threshold
        if residual.dim() == 5 and residual.shape[2] == self.config.K:
            score = residual
        elif residual.dim() == 4:
            score = residual.unsqueeze(2).expand(-1, -1, self.config.K, -1, -1)
        else:
            raise ValueError(f"unexpected residual shape {tuple(residual.shape)}")
        # The contradiction test is *relative to the best explanation available
        # at this location*, because a globally noisy frame would otherwise flag
        # every candidate at once and trigger a pointless full reset.
        best_fit = score.min(dim=2, keepdim=True).values  # [B,V,1,h,w]
        if update_ema and revision is not None:
            with torch.no_grad():
                margin = self.revision.relative_margin
                target = best_fit * margin
                if revision.residual_ema is None or revision.residual_ema.shape != target.shape:
                    revision.residual_ema = target.detach()
                else:
                    alpha = self.revision.residual_ema
                    revision.residual_ema = (
                        alpha * revision.residual_ema + (1.0 - alpha) * target.detach()
                    )
        # Absolute floor from the measurement model: a candidate whose residual is
        # within the declared noise scale is not contradicted, however much worse
        # it looks next to a competitor that happens to fit better.
        floor = torch.full_like(score, threshold)
        relative = best_fit * self.revision.relative_margin
        if revision is not None and revision.residual_ema is not None:
            relative = torch.maximum(relative, revision.residual_ema)
        boundary = torch.maximum(relative, floor)
        contradicted = score > boundary
        # When *every* candidate is worse than the noise floor the measurement
        # itself is suspect (a bad pixel, a mis-registered sample). That is
        # reported through :meth:`all_candidates_bad` rather than silently
        # suppressing the contradiction: suppressing it here would make a real
        # refutation of the leading candidate disappear exactly when the other
        # candidates are also poor.
        return contradicted

    def all_candidates_bad(self, score: torch.Tensor) -> torch.Tensor:
        """``[B,V,1,h,w]`` bool: no candidate explains this measurement.

        A diagnostic for "the measurement is suspect", kept separate from the
        contradiction map so the two claims are never conflated.
        """
        threshold = self.revision.contradiction_threshold
        return (score > threshold).all(dim=2, keepdim=True)

    def withdrawal_energy(
        self,
        score: torch.Tensor,
        support: torch.Tensor,
        *,
        temperature: float,
        semantic_refutation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One observation's supported refutation, separate from measurement NLL.

        No withdrawal when every candidate fits poorly: a reconstruction failure
        alone cannot support a more confident semantic choice. Strong residuals
        reach the declared retention factor; weak ones interpolate continuously.
        The caller freezes this term across solver cycles and duplicate IDs.
        """
        cfg = self.revision
        if not cfg.contradiction_enabled or cfg.downweight_contradicted == 1:
            return score.new_zeros(score.shape[:3])
        if cfg.semantic_refutation:
            if semantic_refutation is None:
                raise ValueError(
                    "semantic refutation must come from candidate measurement comparisons"
                )
            refuted = semantic_refutation
        else:
            refuted = self.contradiction_map(score, update_ema=False)
        justified = refuted & support.bool() & ~self.all_candidates_bad(score)
        strength = (
            (score - cfg.contradiction_threshold)
            / (cfg.strong_contradiction_threshold - cfg.contradiction_threshold)
        ).clamp(0, 1)
        fraction = (strength * justified).sum((-1, -2)) / support.sum((-1, -2)).clamp_min(1)
        return -temperature * math.log(max(cfg.downweight_contradicted, 1e-12)) * fraction

    def semantic_map(self, energy) -> torch.Tensor:
        """Measured preference for a geometrically distinct, well-fitting rival.

        A common prediction error or an unobserved pixel never proves a semantic
        alternative. This predicate uses predictions and actual support only;
        synthetic geometry labels and identifiability flags are never inputs.
        """
        if energy.geometry_gap_from_best is None:
            raise ValueError("semantic comparison needs geometry-bound predictions")
        return (
            energy.support_mask
            & (energy.geometry_gap_from_best >= self.revision.geometry_separation)
            & (energy.local_log_evidence_against >= self.revision.semantic_log_bayes_threshold)
            & (energy.best_normalized_residual <= self.revision.contradiction_threshold)
        ).unsqueeze(1)

    # ----------------------------------------------------------- propagation
    def compatibility(self, dependency_edge: torch.Tensor) -> torch.Tensor:
        """Nonnegative ``[B,V,h,w,4,K,K]`` compatibility; rows normalise at use."""
        # Keep legacy checkpoint forward semantics (negative means absent), but
        # let a negative parameter recover through a surrogate projection gradient.
        return dependency_edge + (dependency_edge.clamp_min(0.0) - dependency_edge).detach()

    def _edge_weights(
        self,
        dependency_edge: torch.Tensor,
        dependency_valid: torch.Tensor,
        *,
        transpose: bool,
    ) -> torch.Tensor:
        """Normalised edge weights with the neighbour axis explicit.

        Returns ``[B,V,h,w,4,K(k),K(l)]`` where ``k`` indexes the candidate at
        ``u`` and ``l`` the candidate at the neighbour ``d``. Rows are normalised
        over ``l`` so a location with more recorded edges does not automatically
        receive a larger message, and off-grid neighbours are zeroed.

        Both messages and refutation read edge[k,l] at the target u: candidate k
        depends on source l at its recorded neighbour. Transposing candidate
        axes does not reverse the spatial edge and is not used for propagation.
        """
        weight = self.compatibility(dependency_edge)
        if transpose:
            weight = weight.transpose(-1, -2)
        batch, views, height, width_px, neighbours, K, _ = weight.shape
        valid = _validity_for(
            dependency_valid, batch, views, height, width_px, neighbours, weight.dtype
        )
        # Row normalisation preserves relative compatibility with different
        # neighbour candidates; sum over source l, then average valid directions.
        row_sum = weight.sum(dim=-1, keepdim=True)
        # Absent rows stay zero in the forward pass, with bounded recovery
        # gradients instead of division by a near-zero denominator.
        denominator = torch.where(row_sum > 1e-8, row_sum, torch.ones_like(row_sum))
        weight = weight / denominator
        weight = weight * valid.view(batch, views, height, width_px, neighbours, 1, 1)
        assert weight.shape == (batch, views, height, width_px, neighbours, K, K), weight.shape
        return weight

    def edge_row_sums(self, dependency_edge: torch.Tensor) -> torch.Tensor:
        """``[B,V,h,w,4,K]`` sum of each dependency row, for the diagnostics.

        These are pre-normalisation diagnostics; messages use relative row weights.
        """
        return dependency_edge.clamp_min(0.0).sum(dim=-1)

    @staticmethod
    def _neighbour_tensor(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Shifted copies of ``value`` along the trailing ``(h, w)`` axes.

        The input is any tensor whose last two axes are the spatial grid,
        ``[..., h, w]``; the result is ``[..., h, w, 4]`` with off-grid neighbours
        zeroed by the boundary-validity mask. The width axis stays in place, so a
        field ``[B,V,K,W,h,w]`` becomes ``[B,V,K,W,h,w,4]``.
        """
        height_px, width_field = value.shape[-2:]
        neighbours = neighbour_stack(value)
        assert neighbours.shape[:-3] == value.shape[:-2], (neighbours.shape, value.shape)
        assert neighbours.shape[-3:] == (height_px, width_field, 4), neighbours.shape
        mask = _validity_for(
            valid, value.shape[0], value.shape[1], height_px, width_field, 4, value.dtype
        )
        # [B,V,h,w,4] -> singleton axes that line up with both the 5-axis
        # (evidence) and 6-axis (field) neighbour layouts.
        while mask.dim() < neighbours.dim():
            mask = mask.unsqueeze(2)
        return neighbours * mask

    def messages(
        self,
        state: torch.Tensor,
        dependency_edge: torch.Tensor,
        dependency_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Neighbourhood message ``[B,V,K,W,h,w]`` over the recorded edges.

        ``dependency_edge[b,v,y,x,d,k,l]`` states that candidate ``k`` at ``u``
        depends on candidate ``l`` at the neighbour ``v_d(u)``. The message a
        location receives is the compatibility-weighted neighbour state,
        row-normalised over ``l`` (see :meth:`_edge_weights`).
        """
        batch, views, K, width, height, width_px = state.shape
        weight = self._edge_weights(dependency_edge, dependency_valid, transpose=False)
        neighbours = self._neighbour_tensor(state, dependency_valid)  # [B,V,K,W,h,w,4]
        # For each dependency (k at u -> l at neighbour d) gather the neighbour
        # state of candidate l with its width axis kept:
        # [B,V,K,W,h,w,4] -> [B,V,h,w,4,K(l),W]
        stacked = neighbours.permute(0, 1, 4, 5, 6, 2, 3).contiguous()
        contribution = weight.unsqueeze(-1) * stacked.unsqueeze(5)  # [..,4,K(k),K(l),W]
        message = contribution.sum(dim=6).sum(dim=4)  # [B,V,h,w,K(target),W]
        degree = dependency_valid.sum(dim=-1).clamp_min(1).unsqueeze(-1).unsqueeze(-1)
        message = message / degree
        return message.permute(0, 1, 4, 5, 2, 3).contiguous()

    def propagate(
        self,
        contradiction: torch.Tensor,
        dependency_edge: torch.Tensor,
        dependency_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate revision priority along the recorded dependency edges.

        Returns ``(propagated_selection, priority)``. ``priority`` accumulates the
        support each (candidate, location) receives from its neighbours (or takes
        its maximum when accumulation is disabled). Selection is the union of
        direct contradiction and per-hop votes above the separately configured
        normalized support threshold, with both vote and threshold decayed once
        per hop. Priority remains an audit of the selected local dependency paths.
        """
        cfg = self.revision
        if not cfg.propagation_enabled or cfg.propagation_hops == 0:
            return contradiction.clone(), contradiction.to(torch.float32)
        batch, views, K, height, width_px = contradiction.shape
        weight = self._edge_weights(dependency_edge, dependency_valid, transpose=False)
        assert weight.shape == (batch, views, height, width_px, 4, K, K), weight.shape
        mask = _validity_for(
            dependency_valid, batch, views, height, width_px, 4, contradiction.dtype
        ).unsqueeze(2)  # [B,V,1,h,w,4]
        source = contradiction.to(torch.float32)
        selection = contradiction.clone()
        priority = source.clone()
        for hop in range(1, cfg.propagation_hops + 1):
            shifted = neighbour_stack(source) * mask  # [B,V,K,h,w,4]
            # Axis roles: ``weight`` is [B,V,h,w,4,K(k),K(l)] where k is the
            # candidate at this location and l the candidate at the neighbour.
            # The vote for candidate k must therefore multiply the *same* k axis:
            # the candidate axis becomes the weight's row axis, not the last data
            # axis, otherwise the dependency table pairs unrelated slots.
            target = shifted.movedim(2, 5).unsqueeze(5)  # [B,V,h,w,4,1,K(l)]
            vote = (weight * target).sum(dim=-1).sum(dim=4).permute(0, 1, 4, 2, 3)
            # Source already carries previous hops' decay: apply it only once.
            vote = vote * cfg.propagation_decay
            priority = priority + vote if cfg.propagate_priority else torch.maximum(priority, vote)
            selection = selection | (vote >= cfg.propagation_threshold * cfg.propagation_decay**hop)
            source = vote
        return selection, priority

    def location_mask(
        self,
        contradiction: torch.Tensor,
        residual_feature: torch.Tensor,
        prior_feature: torch.Tensor,
        location_code: torch.Tensor,
        *,
        cold_start: bool,
    ) -> torch.Tensor:
        """``[B,V,K,h,w]`` float blend in [0,1]: how much of this cycle's update to take.

        The learnable location gate decides how much of the field is worth
        recomposing; where the measurement contradicts a candidate the blend is
        forced to 1 so a refuted cell is always revisited. The gate starts biased
        low (``-2``) so the default is a *local* update rather than a full-field
        rewrite, and its firing rate is reported in the cycle trace as
        ``updated_fraction``.

        A straight-through hard threshold preserves unselected cells exactly while
        allowing gradients into the selection gate during training.
        """
        batch, views = residual_feature.shape[0], residual_feature.shape[1]
        height, width = residual_feature.shape[-2:]
        gate_input = torch.cat(
            [
                residual_feature.mean(dim=2, keepdim=True),
                prior_feature.mean(dim=2, keepdim=True),
                location_code[:, :, :1],
            ],
            dim=2,
        )
        flat = gate_input.reshape(batch * views, gate_input.shape[2], height, width)
        if batch != contradiction.shape[0] or views != contradiction.shape[1]:
            raise ValueError(
                "location gate inputs disagree with the contradiction map: "
                f"gate {(batch, views)} vs contradiction {tuple(contradiction.shape[:2])}"
            )
        probability = torch.sigmoid(self.location_update(flat))
        # [B*V,1,h,w] -> [B,V,1,h,w]: the singleton hypothesis axis must be
        # explicit, otherwise the (K,h,w) tail would broadcast against the map.
        probability = probability.reshape(batch, views, 1, height, width)
        # Straight-through selection learns the gate while leaving unselected
        # cells exactly unchanged in the forward computation.
        hard = (probability >= 0.5).to(probability.dtype)
        selected = hard + probability - probability.detach()
        blend = torch.where(contradiction, torch.ones_like(probability), selected)
        blend = blend.expand(-1, -1, self.config.K, -1, -1)
        if blend.shape != contradiction.shape:
            raise ValueError(
                f"location blend {tuple(blend.shape)} does not match "
                f"contradiction {tuple(contradiction.shape)}"
            )
        cold = torch.as_tensor(cold_start, device=blend.device).bool()
        if cold.ndim:
            cold = cold.reshape(batch, 1, 1, 1, 1)
        return torch.where(cold, torch.ones_like(blend), blend)

    def step(
        self,
        state: HypothesisState,
        normalized_residual: torch.Tensor,
        *,
        residual_feature: torch.Tensor,
        prior_feature: torch.Tensor,
        location_code: torch.Tensor,
        dependency_valid: torch.Tensor | None = None,
        iteration: int = 0,
        cold_start: bool = False,
        measurement_feature: torch.Tensor | None = None,
        semantic_refutation: torch.Tensor | None = None,
    ) -> RevisionOutput:
        """Run one revision cycle and return the new field plus its audit trail."""
        cfg = self.revision
        revision = replace(state.revision)
        cold = torch.as_tensor(cold_start, device=state.h.device).bool()
        if cold.ndim == 0:
            cold = cold.expand(state.h.shape[0])
        if cfg.semantic_refutation:
            if semantic_refutation is None:
                raise ValueError("semantic refutation mask is required")
            contradiction = semantic_refutation
        else:
            contradiction = self.contradiction_map(
                normalized_residual, update_ema=not bool(cold.all()), revision=revision
            )
        if not cfg.contradiction_enabled:
            contradiction = torch.zeros_like(contradiction)
        edge = state.dependency_edge
        if edge is None:
            raise ValueError("dependency edges must be initialised before revision")
        if not cfg.write_dependency_edges:
            edge = torch.zeros_like(edge)
        valid = dependency_valid if dependency_valid is not None else state.dependency_valid
        if valid is None:
            raise ValueError("dependency validity mask is required")
        selection, priority = self.propagate(contradiction, edge, valid)
        fit_correction = torch.zeros_like(selection)
        if cfg.separate_fit_updates:
            fit_correction = normalized_residual > cfg.contradiction_threshold
        update_mask = self.location_mask(
            selection | fit_correction,
            residual_feature,
            prior_feature,
            location_code,
            cold_start=cold_start,
        )
        message = self.messages(state.h, edge, valid)
        candidate = self.cell(
            state.h,
            message,
            residual_feature,
            prior_feature,
            location_code,
            context=None if measurement_feature is None else measurement_feature.unsqueeze(2),
        )
        if candidate.shape != state.h.shape or update_mask.shape[:3] != state.h.shape[:3]:
            raise ValueError(
                "candidate/mask shape mismatch: candidate "
                f"{tuple(candidate.shape)}, mask {tuple(update_mask.shape)}, "
                f"field {tuple(state.h.shape)}"
            )
        # ``blend`` is spatial: the same (candidate, location) decision applies to
        # every channel of that cell, hence the explicit singleton axis. The blend
        # is hard in the forward pass: unselected cells retain their exact value.
        blend = update_mask.unsqueeze(3)
        candidate = state.h + cfg.update_scale * (candidate - state.h)
        updated = blend * candidate + (1.0 - blend) * state.h

        reset = torch.zeros_like(selection)
        if cfg.reset_contradicted:
            reset = self.select_reset(contradiction, selection) & ~cold[:, None, None, None, None]
            reset_mask = (reset & selection).unsqueeze(3)
            if cfg.proposal_enabled:
                # Rebuild the refuted cells from the current measurement evidence.
                proposed = self.proposal(
                    updated
                    if measurement_feature is None
                    else updated + measurement_feature.unsqueeze(2),
                    residual_feature,
                    selection.to(state.h.dtype),
                    location_code,
                )
                updated = torch.where(reset_mask, proposed, updated)
            else:
                # Ablation: the refuted explanation is withdrawn but nothing new is
                # proposed, so those cells stay at their pre-cycle value. They must
                # NOT be left holding the cell update, which would make the switch
                # change cells outside the reset region.
                updated = torch.where(reset_mask, state.h, updated)
        # The controller is stateless with respect to its input: the caller's state
        # object is NOT mutated, so two controllers (or two ablations) can run the
        # same state independently. The new revision bookkeeping travels in
        # ``RevisionOutput.revision`` and the encoder stores it.
        new_revision = RevisionState(
            residual_ema=None if revision.residual_ema is None else revision.residual_ema,
            contradiction=contradiction.detach(),
            contradiction_prev=contradiction.detach(),
            propagated=selection.detach(),
            priority=priority.detach(),
            last_reset=reset.detach(),
        )
        updated_fraction = float(update_mask.to(torch.float32).mean().item())
        fired_fraction = float((update_mask > 0.5).to(torch.float32).mean().item())
        contradicted_fraction = float(contradiction.to(torch.float32).mean().item())
        assert updated.shape == state.h.shape, (updated.shape, state.h.shape)
        return RevisionOutput(
            h=updated,
            priority=priority.mean(dim=2).detach(),
            contradiction=contradiction.detach(),
            propagated=selection.detach(),
            reset=reset.detach(),
            updated_fraction=updated_fraction,
            contradicted_fraction=contradicted_fraction,
            selected_candidates=contradiction.any(dim=(-1, -2)).detach(),
            iteration=iteration,
            converged=False,
            revision=new_revision,
            trace={
                "updated_fraction": updated_fraction,
                "gate_fired_fraction": fired_fraction,
                "contradicted_fraction": contradicted_fraction,
                "reset_fraction": float(reset.to(torch.float32).mean().item()),
                "propagated_fraction": float(selection.to(torch.float32).mean().item()),
                "fit_correction_fraction": float(fit_correction.float().mean()),
            },
        )

    def select_reset(self, contradiction: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
        """Which refuted (candidate, location) cells to rebuild from measurements.

        ``reset_fraction < 1`` keeps only the highest-priority cells, chosen by
        their contradiction priority rank. The choice is deterministic, never a
        random draw: a reset must be reproducible under a fixed seed.
        """
        cfg = self.revision
        base = selection & contradiction
        if cfg.reset_fraction >= 1.0:
            return base
        if cfg.reset_fraction <= 0.0:
            return torch.zeros_like(contradiction)
        batch, views, K, height, width = contradiction.shape
        flat = base.reshape(batch, views, K, height * width)
        count = max(1, int(round(cfg.reset_fraction * flat.shape[-1])))
        keep = flat.topk(count, dim=-1, largest=True).indices
        out = torch.zeros_like(flat)
        out.scatter_(-1, keep, flat.gather(-1, keep))
        return out.reshape(batch, views, K, height, width)


def contradiction_diagnostics(
    output: RevisionOutput,
    weights: torch.Tensor,
) -> dict[str, Any]:
    """Human-readable per-cycle summary used by the training/eval logs."""
    with torch.no_grad():
        top = weights.max(dim=-1).values
        return {
            "iteration": output.iteration,
            "contradicted_fraction": output.contradicted_fraction,
            "updated_fraction": output.updated_fraction,
            "selected_candidates": output.selected_candidates.sum(dim=-1).float().mean().item(),
            "mean_top_weight": float(top.mean().item()),
        }

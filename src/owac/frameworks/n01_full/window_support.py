"""Exact bounded-window support: aligned local GLS and author BOCPD replay.

Every readout starts from one fixed prior and only the retained observations.
The current-reference chart follows actual flow trajectories; the resulting
posterior is a finite-window retrospective model, not an anytime false-alarm bound.
"""

from __future__ import annotations

import torch

from owac.baselines.burst_features import align_features
from owac.frameworks.n01_full.support import (
    SUPPORT_ACCEPTED,
    SUPPORT_DEFERRED,
    SUPPORT_INVALID,
    SUPPORT_RETRACTED,
    SUPPORT_STATE_NAMES,
    SpatialSupport,
    SupportDecision,
    TruncatedBOCPD,
)
from owac.frameworks.n01_full.tensor_gls import fit_aligned_innovations


class WindowGLSSupport(SpatialSupport):
    def _fit(self, difference, mask, variance):
        rows, cols = self.region_grid_shape(*difference.shape[-2:])
        return fit_aligned_innovations(
            difference,
            mask,
            variance,
            rows=rows,
            cols=cols,
            radii=self.config.multiscale_radii,
            affine=self.config.affine,
            minimum_cells=self.config.min_region_cells,
            compatible_score=self.config.large_residual_threshold,
        )

    def replay_window(self, values, cell_mask, source_valid, offsets_xy, variance, visibility=None):
        batch, length, channels, height, width = values.shape
        with torch.no_grad():
            aligned, aligned_mask, offsets = align_features(
                values.detach().float(),
                cell_mask,
                None if offsets_xy is None else offsets_xy.detach(),
            )
            # Unknown interpolation-tap correlations: sum |a| sigma, then square.
            aligned_std, _, _ = align_features(
                variance.detach().clamp_min(1e-12).sqrt(), cell_mask, offsets.detach()
            )
            aligned_variance = aligned_std.square()
            aligned_mask &= source_valid[:, :, None, None, None]
            if visibility is not None:
                aligned_mask &= visibility[:, :, None].to(torch.bool)
            regions = self._region_count(height, width)
            detector = TruncatedBOCPD(self.config, batch, regions, device=values.device)
            # Replay oldest to newest; no posterior from an evicted source survives.
            for slot in range(length - 2, -1, -1):
                valid = aligned_mask[:, slot : slot + 1] & aligned_mask[:, slot + 1 : slot + 2]
                fit = self._fit(
                    aligned[:, slot : slot + 1] - aligned[:, slot + 1 : slot + 2],
                    valid,
                    (
                        aligned_variance[:, slot : slot + 1].sqrt()
                        + aligned_variance[:, slot + 1 : slot + 2].sqrt()
                    ).square(),
                )
                detector.update(fit.score[:, 0], active=fit.valid[:, 0])
            pair_mask = aligned_mask[:, 1:] & aligned_mask[:, :1]
            fit = self._fit(
                aligned[:, 1:] - aligned[:, :1],
                pair_mask,
                (aligned_variance[:, 1:].sqrt() + aligned_variance[:, :1].sqrt()).square(),
            )
            ages = torch.arange(1, length, device=values.device)[None, :, None].expand(
                batch, -1, regions
            )
            # Source-specific posterior: a source from before the current run exits;
            # observations arriving after a change are not mechanically withdrawn too.
            logp = detector.log_message[:, None]
            before_change = detector.run_index[:, None] < ages[..., None]
            neg_inf = torch.full_like(logp, -float("inf"))
            short = torch.logsumexp(torch.where(before_change, logp, neg_inf), -1)
            long = torch.logsumexp(torch.where(~before_change, logp, neg_inf), -1)
            odds = short - long
            odds = torch.nan_to_num(odds, nan=0.0, posinf=100.0, neginf=-100.0)
            large = fit.score >= self.config.large_residual_threshold
            states = torch.where(
                fit.valid,
                torch.full_like(ages, SUPPORT_ACCEPTED),
                torch.full_like(ages, SUPPORT_INVALID),
            )
            # A one-frame anomaly is deferred. Repeated incompatibility in the causal
            # trajectory is required before a source is called structurally withdrawn.
            change = odds >= self.config.retract_logodds
            states = torch.where(
                fit.valid & large, torch.full_like(states, SUPPORT_DEFERRED), states
            )
            compatible_prefix = (fit.valid & ~large).to(torch.int64).cumprod(1).sum(1)
            sustained = compatible_prefix >= max(self.config.defer_observations - 1, 1)
            states = torch.where(
                fit.valid & large & change & sustained[:, None],
                torch.full_like(states, SUPPORT_RETRACTED),
                states,
            )
            usable = self.expand_to_cells(states, source_valid[:, 1:], height, width)
            # Decisions are in the reference chart. They must mask ALIGNED content,
            # not be applied at the same array indices in a different source image.
            decision = SupportDecision(
                state=states,
                source_usable=usable,
                change_logodds=detector.change_logodds(),
                region_residual=fit.score.mean(1),
                age=ages,
                deferred=states == SUPPORT_DEFERRED,
                counts={
                    name: int((states == code).sum()) for code, name in SUPPORT_STATE_NAMES.items()
                },
            )
            state = {
                "detector": detector,
                "age": ages,
                "defer_count": large.to(torch.int64),
                "state": states,
                "chosen_radius": fit.radius,
                "fit_rank": fit.rank,
                "coefficients": fit.coefficients,
                "replay_sources": length,
                "replay_scope": "retained_window_from_fixed_prior",
                "decision_chart": "current_reference_grid",
                "variance_kind": "unknown_correlation_bound_of_declared_domain_variance",
            }
            return state, decision, aligned_mask

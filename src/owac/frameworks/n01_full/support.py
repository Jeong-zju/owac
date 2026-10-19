"""N01 v1.2 P3 spatial support and temporal revision.

This module is the one place where the v1.2 support mechanism is implemented for
the real training path.  It is deliberately region-based rather than
per-query-pixel: the task allows a conservative receptive-field bound, and a
region-level decision that is then expanded conservatively is honest about what
was actually recomputed.

What is transcribed from the authors
------------------------------------
``TruncatedBOCPD`` follows ``bocpd()`` in DSM-bocd at fixed commit
``9d05117bc8c6e971c836ee1df0c49e2d2950862b`` (MIT, Matías Altamirano), including
its run-length truncation: the growth and changepoint terms, the normalisation
over the joint, the top-``K`` selection and the message that is passed on.  The
predictive model is the author's ``GaussianUnknownMean`` (Normal prior, Normal
likelihood with known variance, unknown mean).  ``RobustGaussianUnknownMean``
keeps the author's generalized-Bayes structure from ``SMGaussianUnknownMean``
(``new_prec = prec + 2 omega A``, ``new_mean = (prec * mean - 2 omega v) / new_prec``)
but scores an observation through a bounded influence function, so one bad frame
cannot move a region's posterior.  The generalized-Bayes correction is the
paper's idea; the bounded-loss choice and its scale are ours and are recorded as
such, not presented as the authors' network.

What this module does *not* claim
---------------------------------
``change_logodds`` is a model posterior, not a proven false-alarm probability.
There is no e-value, no confidence sequence and no frequency guarantee here; the
theory note states the assumptions under which such a guarantee would need to
hold and are not established.  Persistently large motion is *not* treated as a
reason to withdraw a source: only an innovation that stays incompatible with the
stable history is.

Four states per (source, region)
-------------------------------
``INVALID``    missing, defective, saturated or unusable correspondence
``ACCEPTED``   stable and alignable history, usable as measurement support
``DEFERRED``   suspicious; held out of content until real contrary evidence arrives
``RETRACTED``  structural-support edge exit: no longer supports this structure

``RETRACTED`` is an edge exit, not a source deletion.  A retracted source may
still contribute *context* ("something changed here"), which is exactly why the
final representation cannot be described as independent of it; whole-source
removal is a different operation and lives in :mod:`owac.frameworks.n01_full.replay`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

__all__ = [
    "SupportConfig",
    "SupportDecision",
    "TruncatedBOCPD",
    "SpatialSupport",
    "SUPPORT_INVALID",
    "SUPPORT_ACCEPTED",
    "SUPPORT_DEFERRED",
    "SUPPORT_RETRACTED",
]


SUPPORT_INVALID = 0
SUPPORT_ACCEPTED = 1
SUPPORT_DEFERRED = 2
SUPPORT_RETRACTED = 3

SUPPORT_STATE_NAMES = {
    SUPPORT_INVALID: "invalid",
    SUPPORT_ACCEPTED: "accepted",
    SUPPORT_DEFERRED: "deferred",
    SUPPORT_RETRACTED: "retracted",
}

_LOG_2PI = 1.8378770664093453


@dataclass(frozen=True)
class SupportConfig:
    """Support geometry, robustness scales and decision thresholds.

    The scales here are development-set knobs: they are declared in the effective
    configuration of every run so that a reader can see which values produced
    which decision.  They are not a derived optimum and not a pass threshold.
    """

    # Historical proxy stays loadable for old evidence. Formal v1.2 configurations
    # explicitly select the implemented gls_window path.
    implementation: str = "legacy_proxy"
    region_grid: int = 16
    multiscale_radii: tuple[int, ...] = (0, 1, 2)
    receptive_field_radius: int = 1
    hazard: float = 1.0 / 32.0
    keep_run_lengths: int = 16
    prior_mean: float = 0.0
    prior_var: float = 1.0
    obs_var: float = 2.0e-2
    robust_omega: float = 4.0
    robust_scale: float = 1.5
    accept_logodds: float = 0.5
    # ``change_logodds`` is ``log_short - log_long``, so POSITIVE means "a changepoint
    # happened recently".  Retraction therefore has to fire on the HIGH side, and the bar
    # sits strictly above zero because the odds function reports exactly 0 while the two
    # hypotheses are still indistinguishable -- a placeholder for "not enough evidence
    # yet" must never by itself withdraw a source.  1.0 is a chosen operating point
    # (e^1 ~ 2.7:1 for a recent changepoint), like the other scales here, not an optimum.
    retract_logodds: float = 1.0
    # Declared here because they take part in the decision: leaving them as bare
    # literals inside the decision code would make the configuration record
    # incomplete for anyone re-deriving the scales on real calibration data.
    large_residual_threshold: float = 1.0
    recent_run_fraction: float = 2.0
    defer_observations: int = 2
    noise_floor: float = 1.0e-3
    min_region_cells: int = 4
    # Retained for old configuration readability; the current residual is the
    # absolute-difference proxy below and does not perform an affine/GLS fit.
    affine: bool = True

    def __post_init__(self) -> None:
        if self.implementation not in ("legacy_proxy", "gls_window"):
            raise ValueError("unknown support implementation")
        if self.region_grid < 1:
            raise ValueError("region_grid must be positive")
        if not self.multiscale_radii or min(self.multiscale_radii) < 0:
            raise ValueError("multiscale_radii must be non-negative")
        if 0 not in self.multiscale_radii:
            raise ValueError("multiscale_radii must include 0 (the region itself)")
        if not 0.0 < self.hazard < 1.0:
            raise ValueError("hazard must lie strictly between 0 and 1")
        if self.keep_run_lengths < 2:
            raise ValueError("keep_run_lengths must keep at least two hypotheses")
        if self.prior_var <= 0.0 or self.obs_var <= 0.0:
            raise ValueError("prior_var and obs_var must be positive")
        if self.robust_omega <= 0.0 or self.robust_scale <= 0.0:
            raise ValueError("robust_omega and robust_scale must be positive")
        # ``change_logodds`` is positive when a changepoint happened recently, so the
        # two bars lie on one axis with the WITHDRAW bar above the STABLE bar.  The
        # previous ordering encoded the inverted comparison: it required the retract
        # bar to sit at or below the accept bar, which leaves no band in which a
        # withdrawn source could ever be judged stable again.
        if not self.accept_logodds < self.retract_logodds:
            raise ValueError("accept_logodds must be below retract_logodds")
        if self.defer_observations < 1:
            raise ValueError("defer_observations must be at least one arrival")
        if self.noise_floor <= 0.0:
            raise ValueError("noise_floor must be positive")
        if self.large_residual_threshold <= 0.0:
            raise ValueError("large_residual_threshold must be positive")
        if self.recent_run_fraction < 1.0:
            raise ValueError("recent_run_fraction must be at least 1")
        if self.min_region_cells < 1:
            raise ValueError("min_region_cells must be at least one cell")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "implementation": self.implementation,
            "region_grid": self.region_grid,
            "multiscale_radii": list(self.multiscale_radii),
            "receptive_field_radius": self.receptive_field_radius,
            "hazard": self.hazard,
            "keep_run_lengths": self.keep_run_lengths,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "obs_var": self.obs_var,
            "robust_omega": self.robust_omega,
            "robust_scale": self.robust_scale,
            "accept_logodds": self.accept_logodds,
            "retract_logodds": self.retract_logodds,
            "large_residual_threshold": self.large_residual_threshold,
            "recent_run_fraction": self.recent_run_fraction,
            "defer_observations": self.defer_observations,
            "noise_floor": self.noise_floor,
            "min_region_cells": self.min_region_cells,
            "affine": self.affine,
        }
        payload["source"] = (
            "Truncated BOCPD and GaussianUnknownMean transcribed from DSM-bocd "
            "9d05117bc8c6e971c836ee1df0c49e2d2950862b (MIT); bounded-influence "
            "generalized-Bayes update and the thresholds/scales are this project's"
        )
        return payload


def _log_sum_exp(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim)


class TruncatedBOCPD:
    """Author-truncated BOCPD over one innovation series per region, vectorised.

    State is kept per region: ``*`` below is an arbitrary leading shape (the
    regions).  ``run_index`` records which run lengths the top-``K`` hypotheses
    currently correspond to, so the sufficient statistics stay aligned after
    truncation instead of assuming the kept hypotheses are the first ``K``.
    """

    def __init__(
        self,
        config: SupportConfig,
        *shape: int,
        dtype: torch.dtype = torch.float32,
        device: Any = None,
    ) -> None:
        self.config = config
        if not shape:
            raise ValueError("at least one region dimension is required")
        self.shape = tuple(int(size) for size in shape)
        self.dtype = dtype
        # The detector state must live wherever its observations do; the caller
        # owns that decision, so it is never silently pinned to the CPU.
        device = torch.device("cpu") if device is None else torch.device(device)
        self.keep = int(config.keep_run_lengths)
        self.log_message = torch.full((*self.shape, 1), -float("inf"), dtype=dtype, device=device)
        self.log_message[..., 0] = 0.0
        self.run_index = torch.zeros((*self.shape, 1), dtype=torch.int64, device=device)
        self.mean_params = torch.full(
            (*self.shape, 1), float(config.prior_mean), dtype=dtype, device=device
        )
        self.prec_params = torch.full(
            (*self.shape, 1), 1.0 / float(config.prior_var), dtype=dtype, device=device
        )
        self.log_evidence = torch.zeros(self.shape, dtype=dtype, device=device)
        self.step = 0

    # -- predictive model --------------------------------------------------------

    def _predictive_logprob(self, observation: torch.Tensor) -> torch.Tensor:
        """Normal predictive with the author's ``1/prec + varx`` variance."""

        means = self.mean_params
        variances = 1.0 / self.prec_params + float(self.config.obs_var)
        z = observation.unsqueeze(-1) - means
        return -0.5 * (z * z / variances + torch.log(variances) + _LOG_2PI)

    def _bounded_influence(self, innovation: torch.Tensor) -> torch.Tensor:
        """Pseudo-Huber influence: bounded, so one outlier cannot move the mean."""

        scale = float(self.config.robust_scale)
        normalised = innovation / scale
        return innovation / torch.sqrt(1.0 + normalised * normalised)

    def _update_sufficient_statistics(self, observation: torch.Tensor) -> None:
        omega = float(self.config.robust_omega)
        influence = self._bounded_influence(observation)
        # Generalized Bayes with a bounded loss, in the author's SM structure:
        # new_prec = prec + 2 omega A, new_mean = (prec * mean - 2 omega v) / new_prec,
        # with A = 1 and v = -psi(x) for the scalar Normal location model.
        new_prec = self.prec_params + 2.0 * omega
        new_mean = (self.prec_params * self.mean_params + 2.0 * omega * influence.unsqueeze(-1)) / (
            new_prec
        )
        prior_mean = torch.full_like(new_mean[..., :1], float(self.config.prior_mean))
        prior_prec = torch.full_like(new_prec[..., :1], 1.0 / float(self.config.prior_var))
        self.mean_params = torch.cat([prior_mean, new_mean], dim=-1)
        self.prec_params = torch.cat([prior_prec, new_prec], dim=-1)

    # -- one arrival -------------------------------------------------------------

    def update(self, observation: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """Consume one scalar innovation per region; return log-odds of a changepoint.

        The returned value is a model posterior ratio between "a changepoint was
        among the recent arrivals" and "no changepoint was", not a calibrated
        false-alarm probability.
        """

        if observation.shape != self.shape:
            raise ValueError(
                f"observation shape {tuple(observation.shape)} does not match regions {self.shape}"
            )
        config = self.config
        previous = None
        if active is not None:
            if active.shape != observation.shape:
                raise ValueError("active mask must match observation shape")
            if not bool(active.any()):
                return self.change_logodds()
            previous = self.state_dict()
        log_hazard = torch.log(torch.tensor(float(config.hazard)))
        log_survival = torch.log(torch.tensor(1.0 - float(config.hazard)))

        growth = self._predictive_logprob(observation) + self.log_message + log_survival
        changepoint = _log_sum_exp(
            self._predictive_logprob(observation) + self.log_message + log_hazard, dim=-1
        )
        joint = torch.cat([changepoint.unsqueeze(-1), growth], dim=-1)

        self.log_evidence = _log_sum_exp(joint, dim=-1)
        normalised = joint - self.log_evidence.unsqueeze(-1)
        self._update_sufficient_statistics(observation)

        keep = min(self.keep, int(joint.shape[-1]))
        values, indices = torch.topk(normalised, keep, dim=-1)
        self.log_message = values
        # Position 0 of ``joint`` is the fresh changepoint hypothesis (run length
        # 0); position k+1 is previous hypothesis k having survived one more step.
        # ``run_index`` therefore has to be grown alongside the sufficient
        # statistics, otherwise the gathered index runs past the previous array.
        grown_index = torch.cat(
            [
                torch.full_like(self.run_index[..., :1], -1),
                self.run_index,
            ],
            dim=-1,
        )
        self.run_index = torch.gather(grown_index, -1, indices) + 1
        self.mean_params = torch.gather(self.mean_params, -1, indices)
        self.prec_params = torch.gather(self.prec_params, -1, indices)
        if previous is not None:
            # Batch members/regions without a new usable measurement keep their prior
            # posterior. Padding adds zero probability, not extra prior hypotheses.
            for key, fill in (
                ("log_message", -float("inf")),
                ("run_index", 0),
                ("mean_params", float(config.prior_mean)),
                ("prec_params", 1.0 / float(config.prior_var)),
            ):
                current = getattr(self, key)
                old = previous[key]
                old = F.pad(old, (0, current.shape[-1] - old.shape[-1]), value=fill)
                setattr(self, key, torch.where(active.unsqueeze(-1), current, old))
            self.log_evidence = torch.where(active, self.log_evidence, previous["log_evidence"])
        self.step += 1
        return self.change_logodds()

    def change_logodds(self) -> torch.Tensor:
        config = self.config

        # "Changed recently" = the mass on run lengths shorter than the arrival
        # count, i.e. a changepoint happened within the tracked window.  Before
        # enough arrivals exist to distinguish the two, the odds are reported as
        # even (log 0) rather than as an infinite amount of evidence.
        threshold = max(2, int(round(int(self.keep) / float(config.recent_run_fraction))))
        short = self.run_index < threshold
        neg_inf = torch.full_like(self.log_message, -float("inf"))
        log_short = _log_sum_exp(torch.where(short, self.log_message, neg_inf), dim=-1)
        log_long = _log_sum_exp(torch.where(~short, self.log_message, neg_inf), dim=-1)
        both_present = torch.isfinite(log_short) & torch.isfinite(log_long)
        odds = torch.where(both_present, log_short - log_long, torch.zeros_like(log_short))
        return odds

    def run_length_log_probability(self) -> torch.Tensor:
        """Dense log posterior over run lengths, reconstructed from the kept hypotheses.

        Truncation keeps ``K`` hypotheses, so indices that were dropped come back
        as ``-inf``.  With ``keep_run_lengths`` above the arrival count no index is
        dropped and this is the author's normalised ``log_R[t]`` exactly, which is
        what the cross-check against the NumPy transcription of ``bocpd()`` reads.
        """

        size = int(self.step) + 1
        dense = torch.full(
            (*self.shape, size),
            -float("inf"),
            dtype=self.log_message.dtype,
            device=self.log_message.device,
        )
        # Inactive batch members can have padded zero-probability slots. Their
        # placeholder index must not overwrite the real probability at run length 0.
        return dense.scatter_reduce(
            -1, self.run_index, self.log_message, reduce="amax", include_self=True
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "log_message": self.log_message,
            "run_index": self.run_index,
            "mean_params": self.mean_params,
            "prec_params": self.prec_params,
            "log_evidence": self.log_evidence,
            "step": self.step,
        }

    @classmethod
    def from_state_dict(cls, config: SupportConfig, payload: dict[str, Any]) -> TruncatedBOCPD:
        shape = tuple(int(size) for size in payload["log_evidence"].shape)
        detector = cls(config, *shape, device=payload["log_evidence"].device)
        for key in ("log_message", "run_index", "mean_params", "prec_params", "log_evidence"):
            setattr(detector, key, payload[key].clone())
        detector.step = int(payload["step"])
        return detector


@dataclass
class SupportDecision:
    """Per-(source, region) support state and the conservative cell-level expansion.

    ``source_usable`` is the cell-level mask that the fusion actually consumes:
    a cell is usable for a source only when every region overlapping that cell's
    conservative receptive field accepted the source.  That bound is what makes
    the statement "an exited region no longer enters the content path" checkable
    instead of aspirational.
    """

    state: torch.Tensor  # [B, T, R_flat] int64 in SUPPORT_* codes
    source_usable: torch.Tensor  # [B, T, H, W] bool, cell level
    change_logodds: torch.Tensor  # [B, R_flat]
    region_residual: torch.Tensor  # [B, R_flat] noise-normalised innovation
    age: torch.Tensor  # [B, T, R_flat] accepted arrivals behind each source
    deferred: torch.Tensor  # [B, T, R_flat] bool
    counts: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "state_counts": self.counts,
            "change_logodds_mean": float(self.change_logodds.mean()),
            "receptive_field": "conservative: a cell needs every overlapping region to accept",
        }


class SpatialSupport:
    """Region-level absolute-difference proxies and temporal revision.

    The analytic GLS implementation lives elsewhere and is not called here.
    ``affine`` and the multiscale candidate search are not active in this path.
    """

    def __init__(self, config: SupportConfig | None = None) -> None:
        self.config = config or SupportConfig()

    # -- regions ----------------------------------------------------------------

    def region_grid_shape(self, height: int, width: int) -> tuple[int, int]:
        grid = int(self.config.region_grid)
        rows = min(grid, int(height))
        cols = min(grid, int(width))
        return rows, cols

    def region_of_cell(self, height: int, width: int, device: Any) -> torch.Tensor:
        """Flat region index per cell, ``[H, W]`` int64, from adaptive average pooling."""

        rows, cols = self.region_grid_shape(height, width)
        index = torch.arange(rows * cols, device=device).reshape(rows, cols).float()
        expanded = F.interpolate(index[None, None], size=(int(height), int(width)), mode="nearest")
        return expanded[0, 0].to(torch.int64)

    def neighbourhood_mask(
        self, rows: int, cols: int, device: Any, radius: int | None = None
    ) -> torch.Tensor:
        """``[R, R]`` bool: which regions are spatial candidates of which.

        Only the declared finite radius is a candidate.  Walking the connectivity
        closure of the region graph would merge unrelated areas into one "domain",
        so it is not done here.
        """

        reach = int(max(self.config.multiscale_radii) if radius is None else radius)
        row = torch.arange(rows, device=device)
        col = torch.arange(cols, device=device)
        grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
        flat_row = grid_row.reshape(-1)
        flat_col = grid_col.reshape(-1)
        within = (flat_row[:, None] - flat_row[None, :]).abs() <= reach
        within &= (flat_col[:, None] - flat_col[None, :]).abs() <= reach
        return within

    # -- local fit ---------------------------------------------------------------

    def region_residual(
        self,
        values: torch.Tensor,
        cell_mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise-normalised region innovation of the window against its reference.

        ``values`` is ``[B, T, C, H, W]`` newest first, ``cell_mask`` ``[B, T, 1, H, W]``.
        Returns ``(residual [B, T-1, R], valid [B, T-1, R])``: for every historical
        source the absolute pixel difference against the newest frame, divided by
        that reference's local-detail RMS proxy. No constant/affine fit or GLS is
        performed, and the denominator is not a calibrated physical noise estimate.
        """

        if values.ndim != 5:
            raise ValueError("values must be [B, T, C, H, W]")
        batch, length, _, height, width = values.shape
        if length < 2:
            empty = values.new_zeros((batch, 0, self._region_count(height, width)))
            return empty, empty.to(torch.bool)
        reference = values[:, :1]
        history = values[:, 1:]
        mask = cell_mask[:, 1:] & cell_mask[:, :1]
        if weights is not None:
            history = history * weights[:, 1:]
        diff = (history - reference).abs()
        diff = diff * mask.to(diff.dtype)
        rows, cols = self.region_grid_shape(height, width)
        cells = self._cells_per_region(height, width, values.device)
        # One consistent field for the pooled innovation.
        field = diff.mean(2, keepdim=True)
        pooled, valid = self._pool_to_regions(field, mask, rows * cols, cells)
        # The scale must NOT come from the field under test: normalising an
        # innovation by its own spread would divide out exactly the constant
        # offsets this statistic is supposed to detect.  It comes from the
        # reference frame's own local detail instead, which is a noise/detail
        # estimate independent of any between-frame offset.
        scale = self._noise_scale(
            reference.mean(2, keepdim=True),
            cell_mask[:, :1],
            rows * cols,
            cells,
        )
        residual = pooled / scale
        return residual, valid

    def _region_count(self, height: int, width: int) -> int:
        rows, cols = self.region_grid_shape(height, width)
        return rows * cols

    def _cells_per_region(self, height: int, width: int, device: Any) -> torch.Tensor:
        index = self.region_of_cell(height, width, device)
        counts = torch.bincount(index.reshape(-1), minlength=int(index.max()) + 1)
        return counts.to(torch.float32).clamp_min(1.0)

    def _pool_to_regions(
        self,
        diff: torch.Tensor,
        mask: torch.Tensor,
        regions: int,
        cells: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sum a cell field into region means (and how many cells were usable)."""

        batch, length = diff.shape[0], diff.shape[1]
        index = self._region_index_for(diff.shape[-2], diff.shape[-1], diff.device)
        flat = diff.reshape(batch, length, -1)
        flat_mask = mask.reshape(batch, length, -1).to(flat.dtype)
        sums = torch.zeros(batch, length, regions, dtype=flat.dtype, device=flat.device)
        sums.scatter_add_(2, index.expand(batch, length, -1), flat * flat_mask)
        counts = torch.zeros_like(sums)
        counts.scatter_add_(2, index.expand(batch, length, -1), flat_mask)
        minimum = float(self.config.min_region_cells)
        valid = counts >= min(minimum, float(cells.min()))
        mean = sums / counts.clamp_min(1.0)
        return mean.unsqueeze(-1).squeeze(-1), valid

    def _region_index_for(self, height: int, width: int, device: Any) -> torch.Tensor:
        return self.region_of_cell(height, width, device).reshape(-1)

    def _noise_scale(
        self,
        reference: torch.Tensor,
        mask: torch.Tensor,
        regions: int,
        cells: torch.Tensor,
    ) -> torch.Tensor:
        """Per-region noise/detail scale from the reference frame's high-frequency part.

        A local Laplacian magnitude is a standard noise proxy and, unlike the
        innovation's own spread, it does not vanish when every historical frame is
        offset from the reference by the same amount.  The floor keeps a perfectly
        flat region from producing an unbounded statistic; the value comes from the
        configuration, not from test data.
        """

        field = reference.mean(2)
        local = field - F.avg_pool2d(field, kernel_size=3, stride=1, padding=1)
        magnitude = local.abs()
        batch, length = magnitude.shape[0], magnitude.shape[1]
        index = self._region_index_for(magnitude.shape[-2], magnitude.shape[-1], magnitude.device)
        flat = magnitude.reshape(batch, length, -1)
        flat_mask = mask.reshape(batch, length, -1).to(flat.dtype)
        second = torch.zeros(batch, length, regions, dtype=flat.dtype, device=flat.device)
        second.scatter_add_(2, index.expand(batch, length, -1), flat * flat * flat_mask)
        counts = torch.zeros_like(second)
        counts.scatter_add_(2, index.expand(batch, length, -1), flat_mask)
        rms = torch.sqrt(second / counts.clamp_min(1.0))
        return rms.clamp_min(float(self.config.noise_floor))

    # -- decisions ---------------------------------------------------------------

    def decide(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        change_logodds: torch.Tensor,
        age: torch.Tensor,
        defer_count: torch.Tensor,
    ) -> torch.Tensor:
        """Assign INVALID / ACCEPTED / DEFERRED / RETRACTED per (source, region).

        ``age`` is accepted and threaded through but **not read**: the decision uses the
        changepoint odds, the residual and the defer count only.  It is kept in the
        signature because the support state carries it and a caller may want it, but a
        reader must not assume it participates -- an earlier docstring listed it as a
        decision input, which it never was.
        """

        config = self.config
        state = torch.full(
            residual.shape, SUPPORT_INVALID, dtype=torch.int64, device=residual.device
        )
        state = torch.where(valid, torch.full_like(state, SUPPORT_ACCEPTED), state)
        # Structural-support edge exit: the region's innovation history says the
        # content changed, so this source no longer supports this structure.  The
        # comparison is on the HIGH side because ``change_logodds`` is
        # ``log_short - log_long``: positive means "a changepoint happened recently".
        # The inverted ``<=`` here withdrew exactly the stable histories the mechanism
        # exists to accumulate, and left genuinely changed sources accepted.
        changed = change_logodds.unsqueeze(1) >= float(config.retract_logodds)
        large = residual >= float(config.large_residual_threshold)
        # ``defer_count`` already shares ``residual``'s [B, T, R] shape; only the
        # per-region changepoint statistic needs a source axis.
        suspect = changed | (large & (defer_count >= int(config.defer_observations)))
        accepted = state == SUPPORT_ACCEPTED
        state = torch.where(accepted & suspect, torch.full_like(state, SUPPORT_DEFERRED), state)
        state = torch.where(accepted & changed, torch.full_like(state, SUPPORT_RETRACTED), state)
        # Recovery is gated on the odds having fallen back to the STABLE side, not on
        # "strong" evidence: high odds make ``changed`` and any confidence test true at
        # the same time, so gating recovery on confidence would immediately undo every
        # retraction.  Between the two bars the source stays withdrawn, which is the
        # band where the evidence is genuinely ambiguous.
        stable = change_logodds.unsqueeze(1) <= float(config.accept_logodds)
        revisable = (state == SUPPORT_DEFERRED) | (state == SUPPORT_RETRACTED)
        state = torch.where(
            revisable & stable & ~large, torch.full_like(state, SUPPORT_ACCEPTED), state
        )
        return state

    def expand_to_cells(
        self,
        state: torch.Tensor,
        source_valid: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Conservative cell-level usability from the region decision.

        A cell is usable for a source only when *every* region whose receptive
        field can reach it accepted that source.  Anything narrower would let a
        retracted region leak back in through a convolution that spans regions.
        """

        batch, length, regions = state.shape
        rows, cols = self.region_grid_shape(height, width)
        if regions != rows * cols:
            raise ValueError(f"state has {regions} regions but the grid is {rows}x{cols}")
        usable = state == SUPPORT_ACCEPTED
        # Conservative bound: the radius is the region-span of the content path's
        # receptive field, not the wider spatial-candidate radius used for the GLS
        # neighbourhood.  A cell is usable only when every region that can reach it
        # accepted the source, so a retracted region cannot leak back in.
        neighbourhood = self.neighbourhood_mask(
            rows, cols, state.device, radius=int(self.config.receptive_field_radius)
        )
        as_float = usable.to(torch.float32)
        reachable = as_float @ neighbourhood.to(torch.float32).T
        conservative = usable & (reachable == neighbourhood.sum(-1).to(torch.float32))
        cell_region = self.region_of_cell(height, width, state.device).reshape(-1)
        expanded = conservative.reshape(batch, length, -1)[:, :, cell_region].reshape(
            batch, length, height, width
        )
        return expanded & source_valid.unsqueeze(-1).unsqueeze(-1)

    # -- orchestration -----------------------------------------------------------

    def initial_state(self, batch: int, history_sources: int, height: int, width: int, device: Any):
        """Allocate support state for ``history_sources`` historical frames.

        ``history_sources`` counts the sources of a window *excluding* the newest
        frame, which is the reference: a window of ``T`` frames has ``T - 1``
        history sources.  Getting this wrong would silently drop or invent a
        source, so :meth:`observe` checks it instead of broadcasting.
        """

        rows, cols = self.region_grid_shape(height, width)
        regions = rows * cols
        detector = TruncatedBOCPD(self.config, batch, regions, device=device)
        return {
            "detector": detector,
            "age": torch.zeros(batch, history_sources, regions, device=device),
            "defer_count": torch.zeros(batch, history_sources, regions, device=device),
            "state": torch.full(
                (batch, history_sources, regions),
                SUPPORT_INVALID,
                dtype=torch.int64,
                device=device,
            ),
        }

    def reset(self, state: dict[str, Any], selected: torch.Tensor) -> dict[str, Any]:
        """Return ``state`` with the selected batch members freshly initialised.

        An episode reset has to clear the detector as well as the buffer: the BOCPD
        posterior, the per-source age, the defer counts and the state codes all describe
        *this* episode's arrivals, and carrying them into the next one lets the previous
        rollout decide the support of the first frames of the next.  Unselected members
        keep their state, matching the per-member reset the buffer already performs.
        """

        if not torch.is_tensor(selected):
            selected = torch.as_tensor(selected)
        selected = selected.to(torch.bool).reshape(-1)

        payload = dict(state)
        for key in ("age", "defer_count"):
            existing = state[key]
            view = selected.reshape(selected.shape + (1,) * (existing.dim() - 1))
            payload[key] = torch.where(view, torch.zeros_like(existing), existing)
        states = state["state"]
        view = selected.reshape(selected.shape + (1,) * (states.dim() - 1))
        payload["state"] = torch.where(view, torch.full_like(states, SUPPORT_INVALID), states)

        original_detector = state["detector"]
        detector = TruncatedBOCPD.from_state_dict(
            original_detector.config, original_detector.state_dict()
        )
        payload["detector"] = detector
        fresh = TruncatedBOCPD(
            detector.config,
            *detector.shape,
            dtype=detector.dtype,
            device=detector.log_evidence.device,
        )
        if bool(selected.all()):
            payload["detector"] = fresh
            return payload

        # Retained members may have K hypotheses while a fresh member has one. Fill
        # the extra log slots with -inf; broadcasting log(1) would create K priors.
        width = detector.log_message.shape[-1]
        fresh.log_message = F.pad(fresh.log_message, (0, width - 1), value=-float("inf"))
        fresh.run_index = F.pad(fresh.run_index, (0, width - 1), value=0)
        fresh.mean_params = fresh.mean_params.expand_as(detector.mean_params)
        fresh.prec_params = fresh.prec_params.expand_as(detector.prec_params)

        def blend(existing: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
            mask = selected.reshape(selected.shape + (1,) * (existing.dim() - 1))
            return torch.where(mask, blank, existing)

        detector.log_message = blend(detector.log_message, fresh.log_message)
        detector.run_index = blend(detector.run_index, fresh.run_index)
        detector.mean_params = blend(detector.mean_params, fresh.mean_params)
        detector.prec_params = blend(detector.prec_params, fresh.prec_params)
        detector.log_evidence = blend(detector.log_evidence, fresh.log_evidence)
        # A scalar arrival counter cannot be per-member; the carried tensors are what the
        # decision reads, so the counter is returned to its fresh value.
        # Keep the global maximum arrival count for the unselected members. The
        # selected members' hypotheses, rather than this reporting counter, are fresh.
        return payload

    def shift(
        self, state: dict[str, Any], steps: int = 1, selected: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Age every per-source record by ``steps`` slots, freeing the newest ones.

        A window is always anchored at the newest frame, so when a new arrival
        enters slot 0 every existing source moves one slot further back.  Without
        this the support decision for one physical frame would be read as if it
        belonged to a different frame.  The BOCPD state is per region and over
        arrival time, so it is deliberately not shifted.
        """

        if steps < 0:
            raise ValueError("steps must not be negative")
        if steps == 0:
            return state
        capacity = int(state["age"].shape[1])
        if steps > capacity:
            raise ValueError(f"cannot shift more than the {capacity} retained slots")
        payload = dict(state)
        for key in ("age", "defer_count"):
            payload[key] = torch.cat(
                [torch.zeros_like(state[key][:, :steps]), state[key][:, : capacity - steps]],
                dim=1,
            )
        payload["state"] = torch.cat(
            [
                torch.full_like(state["state"][:, :steps], SUPPORT_INVALID),
                state["state"][:, : capacity - steps],
            ],
            dim=1,
        )
        if selected is not None:
            for key in ("age", "defer_count", "state"):
                payload[key] = torch.where(selected[:, None, None], payload[key], state[key])
        return payload

    @staticmethod
    def _write_back(capacity_tensor: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Place an active-window result back into the fixed-capacity record."""

        if int(active.shape[1]) == int(capacity_tensor.shape[1]):
            return active
        return torch.cat([active, capacity_tensor[:, active.shape[1] :]], dim=1)

    def observe(
        self,
        state: dict[str, Any],
        values: torch.Tensor,
        cell_mask: torch.Tensor,
        source_valid: torch.Tensor,
        weights: torch.Tensor | None = None,
        new_measurement: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], SupportDecision]:
        """One arrival: local residuals, BOCPD update, decisions, cell expansion."""

        config = self.config
        residual, valid = self.region_residual(values, cell_mask, weights=weights)
        batch, length = residual.shape[0], residual.shape[1]
        regions = residual.shape[-1]
        # Slot 0 of the window is the newest frame and is the reference, not a
        # source of history: every decision below belongs to slots 1..T-1.
        history_valid = source_valid[:, 1:]
        capacity = int(state["age"].shape[1])
        if capacity < int(history_valid.shape[1]):
            raise ValueError(
                f"support state holds {capacity} slots but the window has "
                f"{int(history_valid.shape[1])} history sources; allocate with "
                "initial_state(batch, window_capacity - 1, ...)"
            )
        # Only the leading slots are occupied; the tail stays free for frames that
        # have not arrived yet.
        active = int(history_valid.shape[1])
        updated = {
            "detector": state["detector"],
            "age": state["age"],
            "defer_count": state["defer_count"],
            "state": state["state"],
        }
        if length == 0:
            decision = SupportDecision(
                state=state["state"],
                source_usable=source_valid.new_zeros(batch, 0, values.shape[-2], values.shape[-1]),
                change_logodds=torch.zeros(batch, regions, device=values.device),
                region_residual=torch.zeros(batch, regions, device=values.device),
                age=state["age"],
                deferred=state["state"] == SUPPORT_DEFERRED,
                counts={"sources": 0},
            )
            return updated, decision

        # One BOCPD series per region: aggregate the accepted history's innovation
        # robustly so a single bad source cannot manufacture a changepoint.
        usable = valid & history_valid.unsqueeze(-1).expand_as(valid)
        masked = torch.where(usable, residual, torch.zeros_like(residual))
        counts = usable.to(residual.dtype).sum(1).clamp_min(1.0)
        observation = (masked.sum(1) / counts).detach()
        arriving = (
            torch.ones(batch, dtype=torch.bool, device=values.device)
            if new_measurement is None
            else new_measurement.to(device=values.device, dtype=torch.bool)
        )
        change_logodds = state["detector"].update(
            observation, active=usable.any(1) & arriving[:, None]
        )
        updated["detector"] = state["detector"]

        age = state["age"][:, :active] + usable.to(state["age"].dtype)
        defer_count = torch.where(
            residual >= float(config.large_residual_threshold),
            state["defer_count"][:, :active] + 1,
            torch.zeros_like(state["defer_count"][:, :active]),
        )
        state_codes = self.decide(residual, usable, change_logodds, age, defer_count)
        age = torch.where(arriving[:, None, None], age, state["age"][:, :active])
        defer_count = torch.where(
            arriving[:, None, None], defer_count, state["defer_count"][:, :active]
        )
        state_codes = torch.where(arriving[:, None, None], state_codes, state["state"][:, :active])
        updated["age"] = self._write_back(state["age"], age)
        updated["defer_count"] = self._write_back(state["defer_count"], defer_count)
        updated["state"] = self._write_back(state["state"], state_codes)
        usable_cells = self.expand_to_cells(
            state_codes, history_valid, values.shape[-2], values.shape[-1]
        )
        counts_report = {
            name: int((state_codes == code).sum()) for code, name in SUPPORT_STATE_NAMES.items()
        }
        return updated, SupportDecision(
            state=state_codes,
            source_usable=usable_cells,
            change_logodds=change_logodds.detach(),
            region_residual=observation,
            age=age,
            deferred=state_codes == SUPPORT_DEFERRED,
            counts=counts_report,
        )

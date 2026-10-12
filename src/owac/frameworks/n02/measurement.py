"""Measurement prediction, support masks and the likelihood/energy rules.

The core rule of N02 lives here: a hypothesis is scored by a **robust negative
log-likelihood of the real observations only**. Predictions are compared with
measurements and are never written back as measurements.

Energy conventions (stated once, used everywhere):

* Per observation ``i`` the mean per-valid-pixel Gaussian term is

  ``e_i = mean_valid[ rho(residual) + log(sigma) ]``

  with ``rho`` the Huber-capped squared residual and ``residual = y - mu``. With
  ``robust='gaussian'`` and ``huber_delta`` large enough this is exactly
  ``0.5 * residual^2 / variance + log(sigma)``, the Gaussian negative
  log-likelihood up to the constant ``0.5*log(2*pi)``, which cancels in the
  softmax over candidates.
* ``measurement_energy_k = sum_i mask_i * e_i`` over the *frozen* measurement
  set. Because each ``e_i`` is already a per-pixel mean, the sum is comparable
  across frames with different resolutions; it still scales with the number of
  retained frames, which is why the buffer's history scheme is declared in the
  config instead of being hidden inside a normalisation factor.
* ``energy_k = measurement_energy_k + beta_prior * prior_energy_k
  + beta_relation * relation_energy_k`` and ``weights = softmax(-energy / T)``.

For a native CFA measurement the head is only evaluated at the actually sampled
coordinates and filter responses; for the three-channel pseudo-RAW input mode a
separate declared head predicts the three channels. The two modes never share a
head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from owac.frameworks.n02.config import MeasurementConfig


def _group_count(channels: int, width: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and width % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    """Small residual conv block with GroupNorm; input/output width preserved."""

    def __init__(self, width: int, kernel: int = 3) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv2d(width, width, kernel, padding=padding)
        self.norm1 = nn.GroupNorm(_group_count(width, 1), width)
        self.conv2 = nn.Conv2d(width, width, kernel, padding=padding)
        self.norm2 = nn.GroupNorm(_group_count(width, 1), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class RAWStem(nn.Module):
    """Measurement-only stem: raw values plus masks/timing to a feature grid.

    ``metadata`` channels are only those the caller declared available; the
    stem never receives a hidden scene parameter.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        layers: int = 2,
        kernel: int = 3,
        metadata_channels: int = 4,
    ) -> None:
        super().__init__()
        self.metadata_channels = metadata_channels
        total = in_channels * 3 + metadata_channels
        self.input = nn.Conv2d(total, width, kernel, padding=kernel // 2)
        self.blocks = nn.ModuleList(ConvBlock(width, kernel) for _ in range(max(1, layers - 1)))
        self.width = width

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        bad: torch.Tensor,
        saturated: torch.Tensor,
        metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        channels = values.shape[1]
        if valid.shape[1] != channels:
            valid = valid.expand(-1, channels, -1, -1)
        if bad.shape[1] != channels:
            bad = bad.expand(-1, channels, -1, -1)
        if saturated.shape[1] != channels:
            saturated = saturated.expand(-1, channels, -1, -1)
        pieces = [
            values * valid,
            valid.float(),
            bad.float() + saturated.float(),
        ]
        if self.metadata_channels:
            if metadata is None:
                meta = values.new_zeros(
                    values.shape[0], self.metadata_channels, values.shape[-2], values.shape[-1]
                )
            else:
                meta = metadata
                if meta.shape[-2:] != values.shape[-2:]:
                    meta = F.interpolate(
                        meta, size=values.shape[-2:], mode="bilinear", align_corners=False
                    )
            pieces.append(meta)
        x = torch.cat(pieces, dim=1)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return x


@dataclass(slots=True)
class MeasurementPrediction:
    """A hypothesis's prediction of the measurement set it is scored against."""

    mean: torch.Tensor  # [B, V, K, C, h, w]
    variance: torch.Tensor  # [B, V, K, C, h, w], same units squared
    geometry: torch.Tensor  # [B, V, K, geometry_dim] declared latent geometry
    predictions_at_samples: torch.Tensor | None = None  # [B, V, K, T, C, N]
    sample_index: torch.Tensor | None = None  # [B, V, T, C, N, 2] (y, x) long


class MeasurementHead(nn.Module):
    """Per-mode raw-value head. ``bayer_cfa`` uses one RAW channel, RGB3 uses three."""

    def __init__(
        self,
        width: int,
        K: int,
        out_channels: int,
        *,
        predict_variance: bool = True,
        geometry_dim: int = 8,
        kernel: int = 3,
    ) -> None:
        super().__init__()
        self.K = K
        self.out_channels = out_channels
        self.predict_variance = predict_variance
        self.geometry_dim = geometry_dim
        hidden = max(8, width // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(width * K, hidden * K, kernel, padding=kernel // 2, groups=K),
            nn.GroupNorm(_group_count(hidden * K, K), hidden * K),
            nn.SiLU(),
            nn.Conv2d(hidden * K, hidden * K, kernel, padding=kernel // 2, groups=K),
            nn.GroupNorm(_group_count(hidden * K, K), hidden * K),
            nn.SiLU(),
        )
        self.value = nn.Conv2d(hidden * K, out_channels * K, 1, groups=K)
        self.log_variance = (
            nn.Conv2d(hidden * K, out_channels * K, 1, groups=K) if predict_variance else None
        )
        self.geometry = nn.Sequential(
            nn.Linear(hidden, max(8, hidden)),
            nn.SiLU(),
            nn.Linear(max(8, hidden), geometry_dim),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``features`` is ``[B*V, K, D', h, w]``; returns mean/log-variance/geometry."""
        batch_views, K, width, height, width_px = features.shape
        x = features.reshape(batch_views, K * width, height, width_px)
        x = self.trunk(x)
        hidden = x.shape[1] // K
        mean = self.value(x).reshape(batch_views, K, self.out_channels, height, width_px)
        if self.log_variance is None:
            log_variance = torch.zeros_like(mean)
        else:
            log_variance = self.log_variance(x).reshape_as(mean).clamp(-12.0, 12.0)
        pooled = F.adaptive_avg_pool2d(x, 1).reshape(batch_views, K, hidden)
        geometry = self.geometry(pooled)
        return mean, log_variance, geometry


def sample_at_coordinates(
    mean: torch.Tensor,
    coordinates: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bilinear gather of predictions at real sampling coordinates.

    ``mean`` is ``[N, C, h, w]``, ``coordinates`` is ``[N, C, n, 2]`` holding
    ``(y, x)`` in pixel units of the prediction grid. Returns ``[N, C, n]``.

    This is the native-CFA path: the head is only ever evaluated where the
    sensor actually sampled a filtered response, so no demosaicing is implied.
    """
    if coordinates.shape[0] != mean.shape[0]:
        raise ValueError(
            f"coordinate batch {coordinates.shape[0]} does not match predictions {mean.shape[0]}"
        )
    samples, channels, points, _ = coordinates.shape
    if channels != mean.shape[1]:
        raise ValueError("coordinate channels must match prediction channels")
    height, width = mean.shape[-2:]
    base = coordinates - coordinates.new_tensor([0.0, 0.0])
    y = base[..., 0]
    x = base[..., 1]
    if height > 1:
        norm_y = 2.0 * y / max(height - 1, 1) - 1.0
    else:
        norm_y = torch.zeros_like(y)
    if width > 1:
        norm_x = 2.0 * x / max(width - 1, 1) - 1.0
    else:
        norm_x = torch.zeros_like(x)
    grid = torch.stack([norm_x, norm_y], dim=-1)  # [N, C, n, 2]
    grid = grid.reshape(samples * channels, points, 1, 2)
    gathered = F.grid_sample(
        mean.reshape(samples * channels, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    gathered = gathered.reshape(samples, channels, points)
    if sample_weights is not None:
        gathered = gathered * sample_weights
    return gathered


def gaussian_nll(
    residual: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    robust: str = "huber",
    huber_delta: float = 3.0,
) -> torch.Tensor:  # noqa: D401 - see the module docstring for the convention
    """Gaussian NLL ``0.5 * r^2 / sigma^2 + log(sigma)`` with an optional Huber cap.

    The Huber cap limits how much a single outlier (a bad pixel, a specular
    spike, a mis-registered sample) can dominate the hypothesis score. The
    ``log variance`` term is kept outside the cap so the predicted scale is
    still regularised by the likelihood itself.
    """
    variance = torch.exp(log_variance).clamp_min(1.0e-12)
    normalized = residual / torch.sqrt(variance)
    if robust == "huber":
        absolute = normalized.abs()
        delta = float(huber_delta)
        quadratic = 0.5 * normalized**2
        linear = delta * (absolute - 0.5 * delta)
        rho = torch.where(absolute <= delta, quadratic, linear)
    else:
        rho = 0.5 * normalized**2
    # Gaussian NLL: 0.5 * r^2 / sigma^2 + log(sigma). With log_variance = log(sigma^2),
    # log(sigma) = 0.5 * log_variance. (An earlier version added 0.5*log_variance
    # here, i.e. 0.25*log(sigma^2), which under-weighted the scale term; the exact
    # analytic reference in owac.frameworks.n02.analytic now pins this.)
    return rho + 0.5 * log_variance


@dataclass(slots=True)
class EnergyResult:
    """Energy of every hypothesis against the frozen measurement set."""

    energy: torch.Tensor  # [B, V, K]
    measurement_energy: torch.Tensor  # [B, V, K], same value before prior terms
    normalized_residual: torch.Tensor  # [B, V, K, h, w] pooled |r|/sigma for the map
    per_observation: list[torch.Tensor] = field(default_factory=list)
    observations_used: int = 0
    duplicates_skipped: int = 0
    effective_samples: float = 0.0
    unresolved: bool = False
    diagnostics: dict = field(default_factory=dict)


def compute_measurement_energy(
    prediction: MeasurementPrediction,
    buffer: Any,
    config: MeasurementConfig,
    *,
    view_index: int,
) -> EnergyResult:
    """Sum robust NLL of every unique buffered observation for each hypothesis.

    ``prediction.mean``/``variance`` are ``[B, V, K, C, h, w]``. Each buffer
    frame is compared once: the frame's own ``new_measurement`` decision was made
    when it entered the buffer, so an internal cycle can never add the same
    likelihood twice.
    """
    mean = prediction.mean[:, view_index]
    variance = prediction.variance[:, view_index]
    batch, K, channels, height, width = mean.shape
    device = mean.device
    energy = torch.zeros(batch, K, device=device, dtype=mean.dtype)
    residual_map = torch.zeros(batch, K, height, width, device=device, dtype=mean.dtype)
    per_observation: list[torch.Tensor] = []
    effective = 0.0
    frames = buffer.frames
    for frame in frames:
        values = frame.values
        if values.shape[0] != batch:
            if values.shape[0] == 1:
                values = values.expand(batch, -1, -1, -1)
            else:
                raise ValueError("buffer batch size does not match prediction batch size")
        if values.shape[1] != channels:
            raise ValueError(
                f"measurement has {values.shape[1]} channels, head predicts {channels}; "
                "the CFA and three-channel modes must use separate heads"
            )
        if values.shape[-2:] != (height, width):
            values = F.interpolate(
                values, size=(height, width), mode="bilinear", align_corners=False
            )
        valid = frame.valid_mask
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        elif valid.shape[1] != channels:
            valid = valid.expand(-1, channels, -1, -1)
        if valid.shape[-2:] != (height, width):
            valid = F.interpolate(valid.float(), size=(height, width), mode="nearest") > 0.5
        observed = valid.clone()
        if not config.saturation_is_observation and frame.saturated_mask is not None:
            saturated = frame.saturated_mask
            if saturated.shape[1] != channels:
                saturated = saturated.expand(-1, channels, -1, -1)
            if saturated.shape[-2:] != (height, width):
                saturated = (
                    F.interpolate(saturated.float(), size=(height, width), mode="nearest") > 0.5
                )
            observed = observed & ~saturated
        if frame.bad_mask is not None:
            bad = frame.bad_mask
            if bad.shape[1] != channels:
                bad = bad.expand(-1, channels, -1, -1)
            if bad.shape[-2:] != (height, width):
                bad = F.interpolate(bad.float(), size=(height, width), mode="nearest") > 0.5
            observed = observed & ~bad
        log_variance = torch.log(variance.clamp_min(config.variance_floor))
        residual = values.unsqueeze(1) - mean  # [B, K, C, h, w]
        nll = gaussian_nll(
            residual, log_variance, robust=config.robust, huber_delta=config.huber_delta
        )
        weights = observed.unsqueeze(1).to(nll.dtype)  # [B, 1, C, h, w]
        count = weights.sum(dim=(2, 3, 4)).clamp_min(1.0)  # [B, 1]
        per_frame = (nll * weights).sum(dim=(2, 3, 4)) / count  # [B, K]
        energy = energy + per_frame
        per_observation.append(per_frame.detach())
        effective += float(count.mean().item())
        with torch.no_grad():
            normalize = torch.sqrt(variance.clamp_min(config.variance_floor))
            normalized = (residual / normalize).abs()  # [B, K, C, h, w]
            pooled = (normalized * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            residual_map = residual_map + pooled.detach()
    weight_map = None
    if frames:
        weight_map = torch.full_like(residual_map, 1.0 / len(frames))
        residual_map = residual_map * weight_map
    duplicate_count = int(getattr(buffer, "duplicate_count", 0))
    return EnergyResult(
        energy=energy,
        measurement_energy=energy,
        normalized_residual=residual_map,
        per_observation=per_observation,
        observations_used=len(frames),
        duplicates_skipped=duplicate_count,
        effective_samples=effective,
        diagnostics={
            "observations": len(frames),
            "duplicates_skipped": duplicate_count,
            "history_len": len(frames),
            "effective_samples": effective,
        },
    )


class MeasurementModel(nn.Module):
    """Ties the stem, the per-mode heads and the energy computation together.

    The module exposes exactly what the N02 loop needs and nothing that could
    let a prediction leak into the buffer.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        measurement = config.measurement
        self.input_mode = measurement.input_mode
        self.out_channels = 1 if self.input_mode == "bayer_cfa" else 3
        width = config.stem_width
        self.stem = RAWStem(
            in_channels=self.out_channels,
            width=width,
            layers=config.stem_layers,
            kernel=config.cell_kernel,
            metadata_channels=4,
        )
        self.head = MeasurementHead(
            width=width,
            K=config.K,
            out_channels=self.out_channels,
            predict_variance=measurement.predict_variance,
            geometry_dim=measurement.geometry_dim,
            kernel=config.cell_kernel,
        )
        self.geometry_dim = measurement.geometry_dim

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        bad: torch.Tensor,
        saturated: torch.Tensor,
        metadata: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.stem(values, valid, bad, saturated, metadata)

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``features`` ``[B, V, K, D, h, w]`` -> mean/log-variance/geometry."""
        batch, views, K, width, height, width_px = features.shape
        flat = features.reshape(batch * views, K, width, height, width_px)
        mean, log_variance, geometry = self.head(flat)
        mean = mean.reshape(batch, views, K, self.out_channels, height, width_px)
        log_variance = log_variance.reshape_as(mean)
        geometry = geometry.reshape(batch, views, K, self.geometry_dim)
        return mean, log_variance, geometry

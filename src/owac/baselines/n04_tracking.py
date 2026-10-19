"""Track identical instantaneous positions; labels never enter these interfaces.

Positions use one camera-compensated coordinate frame and pixel units. The caller
converts outputs to current-camera coordinates with the same online pose for all
methods. Missing measurements are explicitly masked before any state update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._vendor.rkn.cell import RKNCell


@dataclass
class TrackingOutput:
    position: torch.Tensor
    variance: torch.Tensor | None = None
    prior: torch.Tensor | None = None


def hold_last(position: torch.Tensor, valid: torch.Tensor) -> TrackingOutput:
    """Unfiltered visible observations; hold last position through missing frames."""
    last = torch.zeros_like(position[0])
    outputs = []
    for measurement, available in zip(position, valid, strict=True):
        last = torch.where(available[:, None], measurement, last)
        outputs.append(last)
    return TrackingOutput(torch.stack(outputs))


class RKNPositionTracker(nn.Module):
    """Author RKN cell around a fixed 2-D observation, with no learned input projection.

    The cell performs current update then next prediction. We return the current
    posterior, feeding only the *next* prior into the next iteration. First valid
    measurement initializes position; the unobserved latent half starts at zero.
    """

    def __init__(self, measurement_variance: torch.Tensor, *, scale: float = 32.0):
        super().__init__()
        self.scale = float(scale)
        self.register_buffer("observation_variance", measurement_variance.float() / scale**2)
        config = RKNCell.get_default_config()
        config.never_invalid = False
        config.num_basis = 5
        config.bandwidth = 0
        config.trans_net_hidden_units = [16]
        config.trans_covar = max(float(self.observation_variance.mean()) * 0.05, 1e-7)
        config.finalize_modifying()
        self.cell = RKNCell(2, config)

    def forward(self, position: torch.Tensor, valid: torch.Tensor) -> TrackingOutput:
        observations = torch.where(valid[..., None], position / self.scale - 1.0, 0.0)
        b = position.shape[1]
        prior = position.new_zeros(b, 4)
        covariance = [position.new_ones(b, 2), position.new_ones(b, 2), position.new_zeros(b, 2)]
        started = torch.zeros(b, dtype=torch.bool, device=position.device)
        means, variances, priors = [], [], []
        for observation, available in zip(observations, valid, strict=True):
            initialize = available & ~started
            initial_mean = torch.cat((observation, torch.zeros_like(observation)), -1)
            prior = torch.where(initialize[:, None], initial_mean, prior)
            started = started | available
            priors.append((prior[:, :2] + 1.0) * self.scale)
            posterior, post_cov, prior, covariance = self.cell(
                prior, covariance, observation, self.observation_variance.expand(b, -1), available
            )
            means.append((posterior[:, :2] + 1.0) * self.scale)
            variances.append(post_cov[0] * self.scale**2)
        return TrackingOutput(torch.stack(means), torch.stack(variances), torch.stack(priors))

"""Independent tiny prefix likelihood-ratio scorer; no N02 revision integration.

The observation embedding contains only frozen RAW-stem features. The two
declared hypotheses mean opening-left/right (-1/+1), not learned slot IDs.
An upstream NRE loss receives independent episodes on its batch axis; time
remains inside each episode. Scores describe whole causal prefixes, so a new
prefix REPLACES earlier evidence rather than being multiplied into it.
"""

from __future__ import annotations

import torch
from torch import nn


class PrefixRatioHead(nn.Module):
    """Small spatial projection + causal GRU + shared candidate-conditioned MLP."""

    def __init__(self, width=32, grid=16, hidden=32):
        super().__init__()
        self.width, self.grid, self.hidden = width, grid, hidden
        self.spatial = nn.Sequential(
            nn.Conv2d(width, 4, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(4 * grid * grid, hidden),
            nn.Tanh(),
        )
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.ratio = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, theta, features):
        """theta [*, B, 1], features [B,T,D,H,W] -> log-ratio [*,B,T].

        No prior, hidden renderer state, target template, cue mask, episode ID,
        or time-dependent label is accepted by this interface.
        """
        if features.ndim != 5 or features.shape[2:] != (self.width, self.grid, self.grid):
            raise ValueError("expected [B,T,width,grid,grid] frozen RAW features")
        if features.requires_grad:
            raise ValueError("this comparison freezes the RAW stem")
        if theta.shape[-1] != 1:
            raise ValueError("one physical binary hypothesis coordinate is required")
        batch, steps = features.shape[:2]
        spatial = self.spatial(features.reshape(-1, self.width, self.grid, self.grid))
        prefix, _ = self.temporal(spatial.reshape(batch, steps, self.hidden))
        candidates, prefix = torch.broadcast_tensors(theta.unsqueeze(-2), prefix)
        return self.ratio(torch.cat((candidates[..., :1], prefix), -1)).squeeze(-1)

    def scores(self, features):
        """Return [B,T,2] scores for the same declared pair in every episode."""
        theta = features.new_tensor([-1.0, 1.0]).reshape(2, 1, 1)
        return self(theta, features).permute(1, 2, 0)


def posterior_from_prefix(scores, prior_p1=0.5):
    """Combine initial prior ONCE with prefix evidence; never sum prefix scores.

    This yields a learned posterior approximation, not an absolute fit score or
    a local semantic refutation map. No special no-cue/label oracle is used.
    """
    if scores.shape[-1] != 2 or not torch.isfinite(scores).all():
        raise ValueError("expected finite binary scores")
    prior = torch.as_tensor(prior_p1, device=scores.device, dtype=scores.dtype)
    if not torch.all((prior > 0) & (prior < 1)):
        raise ValueError("prior must lie strictly between zero and one")
    return torch.sigmoid(torch.logit(prior) + scores[..., 1] - scores[..., 0])

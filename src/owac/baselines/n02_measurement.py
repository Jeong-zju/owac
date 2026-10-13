"""Retrainable RAW measurement classifier with a maintained upstream embedding.

Cross-entropy with balanced, paired worlds learns p(theta|causal RAW prefix).
Subtracting the declared TRAINING class log-prior yields relative likelihood
scores. External test priors enter once, downstream. This is a discriminative
measurement baseline, not an unchanged NRE/BNRE algorithm or an absolute density.
"""

from __future__ import annotations

import torch
from torch import nn

from owac.baselines.sbi_embedding import CNNEmbedding


class RetrainedMeasurementScorer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean).float().reshape(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).float().reshape(1, 1, 3, 1, 1))
        if (
            not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or not (self.std > 0).all()
        ):
            raise ValueError("training-only normalization must be finite and positive")
        self.embedding = CNNEmbedding(
            input_shape=(64, 64),
            in_channels=3,
            out_channels_per_layer=[6, 12],
            num_conv_layers=2,
            num_linear_layers=2,
            num_linear_units=32,
            output_dim=32,
            kernel_size=5,
            pool_kernel_size=2,
        )
        self.temporal = nn.GRU(32, 32, batch_first=True)
        self.classifier = nn.Linear(32, 2)
        self.register_buffer("training_log_prior", torch.full((2,), 0.5).log())

    def forward(self, raw):
        if raw.ndim != 5 or raw.shape[2:] != (3, 64, 64):
            raise ValueError("expected [B,T,3,64,64] pseudo-RAW, without extra target channels")
        if not torch.isfinite(raw).all():
            raise ValueError("this synthetic protocol requires fully valid finite measurements")
        batch, steps = raw.shape[:2]
        observed = ((raw - self.mean) / self.std).reshape(-1, 3, 64, 64)
        embedded = self.embedding(observed).reshape(batch, steps, 32)
        prefix, _ = self.temporal(embedded)
        return self.classifier(prefix)

    def scores(self, raw):
        # Softmax normalization adds a shared offset; logits minus class prior
        # produce the same relative ratio without needless log_softmax rounding.
        return self(raw) - self.training_log_prior

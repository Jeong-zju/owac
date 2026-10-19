"""Optional measurement-content path, separate from support and selection.

The original integrator remains the definition of source weights, source identity,
and fractional interpolation. Shared-grid channels are integrated in separate
views using that same primitive; CFA planes already have separate statistics.
The resulting fields are internal measured descriptors, never a reconstructed
RGB image. Reliability is explicit and is not inferred from learned feature norm.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .config import N01Config
from .support import SupportDomainIntegrator


@dataclass
class ContentFields:
    """Per-measurement-channel fields [B,C,H,W] before the feature update.

    ``variance_proxy`` averages the overlapping candidate variance proxies with
    nonnegative normalized weights. This is a conservative mixture noise proxy
    under the source independence assumptions; it is not a calibrated posterior
    and does not model data-dependent selection uncertainty.
    """

    mean: torch.Tensor
    variance_proxy: torch.Tensor
    age: torch.Tensor
    valid: torch.Tensor
    coordinates: torch.Tensor  # [B,C,2,H,W], actual sampling phase coordinates


def sampling_coordinates(config, sample_xy, reference):
    """Keep channel identity and measured phase offsets without CFA averaging."""
    if sample_xy is None:
        sample_xy = (
            [(0, 0), (0, 1), (1, 0), (1, 1)] if config.planes == 4 else [(0, 0)] * config.channels
        )
    coordinates = torch.as_tensor(sample_xy, device=reference.device, dtype=reference.dtype)
    if coordinates.ndim == 2:
        coordinates = coordinates[None].expand(reference.shape[0], -1, -1)
    if coordinates.shape != (reference.shape[0], config.channels, 2):
        raise ValueError("sample_xy must be [C,2] or [B,C,2]")
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError("sample_xy must contain finite measured coordinates")
    return coordinates[:, :, :, None, None].expand(-1, -1, -1, *reference.shape[-2:])


def content_fields(
    config: N01Config,
    state,
    statistics,
    selection,
    sample_xy=None,
    *,
    integrator: SupportDomainIntegrator | None = None,
    edges=None,
    motion=None,
    estimator=None,
) -> ContentFields:
    """Build content using the already selected domains; never change selection.

    For shared-grid multiple channels, ``integrator`` is the unchanged primitive
    configured for one channel, with no extra learned support parameters. Every
    channel receives exactly the same support graph and candidate selection.
    The legacy shared-grid selector requires all channels to be valid for its
    own support statistics. Preserving its semantics means that missing-channel
    history can remain unselected even if another channel has usable sources.
    """
    if config.content_mode == "current_control":
        mean, variance = state.values[0], state.variance[0]
        valid = state.valid[0] & torch.isfinite(mean) & torch.isfinite(variance)
        age = torch.zeros_like(mean)
    elif config.content_mode == "integrated":
        channel_statistics = []
        if config.planes == 4 or config.channels == 1:
            channel_statistics.append(statistics)
        else:
            if integrator is None or edges is None:
                raise ValueError("shared-grid content requires its source integrator and edges")
            for channel in range(config.channels):
                values = state.values[:, :, channel : channel + 1]
                variance = state.variance[:, :, channel : channel + 1]
                valid = state.valid[:, :, channel : channel + 1]
                valid = valid & torch.isfinite(values) & torch.isfinite(variance)
                view = replace(
                    state,
                    config=integrator.config,
                    values=values.masked_fill(~valid, 0.0),
                    variance=variance.masked_fill(~valid, 1.0),
                    valid=valid,
                )
                result, _ = integrator.domain_statistics(
                    view, edges, motion=motion, estimator=estimator
                )
                channel_statistics.append(result)

        def stack(name):
            return torch.cat([getattr(item, name) for item in channel_statistics], dim=2)[
                selection["slice"]
            ]

        means, variances, ages = stack("mu"), stack("variance_proxy"), stack("age")
        available = (
            (stack("support_count") > 0)
            & torch.isfinite(means)
            & torch.isfinite(variances)
            & torch.isfinite(ages)
        )
        weights = selection["weights"] * available
        total = weights.sum(0)
        valid = total > 0
        weights = weights / total.clamp_min(1e-12)[None]

        def mix(tensor):
            return (weights * tensor.masked_fill(~available, 0.0)).sum(0)

        mean, variance, age = mix(means), mix(variances), mix(ages)
    else:
        raise ValueError("content_fields requires an enabled content_mode")
    coordinates = sampling_coordinates(config, sample_xy, mean)
    return ContentFields(
        mean=mean.masked_fill(~valid, 0.0),
        variance_proxy=variance.masked_fill(~valid, 0.0),
        age=age.masked_fill(~valid, 0.0),
        valid=valid,
        coordinates=coordinates.masked_fill(~valid[:, :, None], 0.0),
    )


class ContentUpdate(nn.Module):
    """Measured content enters proposal; quality alone sets channel reliability.

    A pointwise projection preserves distinct measured channels and phases before
    mixing into features. No bias or spatial interpolation can inject content at
    a site where all channels lack support. The ordinary EvidenceBlock performs
    the existing spatial feature update after adding this measured contribution.
    """

    def __init__(self, config: N01Config):
        super().__init__()
        self.reliability = nn.Conv2d(
            3 * config.channels, config.channels, 1, groups=config.channels
        )
        self.projection = nn.Sequential(
            nn.Conv2d(3 * config.channels, config.width, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(config.width, config.width, 1, bias=False),
        )

    def forward(self, content: ContentFields, scales):
        valid = content.valid

        def bounded(value, scale):
            value = value.masked_fill(~valid, 0.0)
            return value / (scale.clamp_min(1e-6) + value.abs())

        mean = bounded(content.mean, scales[5])
        variance = bounded(content.variance_proxy, scales[0])
        age = bounded(content.age, scales[1])
        quality = torch.stack([variance, age, valid.float()], dim=2).flatten(1, 2)
        reliability = torch.sigmoid(self.reliability(quality)) * valid
        coordinates = content.coordinates.masked_fill(~valid[:, :, None], 0.0)
        measured = torch.cat([mean[:, :, None], coordinates], dim=2)
        measured = (measured * reliability[:, :, None]).flatten(1, 2)
        return self.projection(measured) * valid.any(1, keepdim=True)

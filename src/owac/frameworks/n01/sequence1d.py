"""One-dimensional heteroscedastic sequence adapter for N01.

The implementation prompt asks for a second input structure that reuses the same
support-domain integration and split rules, "to verify that they do not depend on
two-dimensional convolution to be defined".  This module provides it:

* the measurement is a one-dimensional field of ``L`` samples per frame, each with
  its own declared noise variance (heteroscedastic along the sequence);
* the same :class:`~owac.frameworks.n01.state.N01State` buffer and the same
  :class:`~owac.frameworks.n01.support.SupportDomainIntegrator` compute the support
  graph, the merge/split hysteresis and every per-source statistic.  The sequence
  is presented on a degenerate ``1 x L`` plane grid, so the row direction simply
  contributes no support (its path weights are exactly zero) and the support
  relations reduce to the one-dimensional ones;
* only the input stem, the evidence blocks and the readout are one-dimensional
  (``Conv1d``/``GroupNorm``), because a ``3x3`` convolution is not the rule under
  test.

The core rules kept unchanged: unique-frame buffer, ``w = m * c / (v + eps)``,
hop-bounded path weights, ``merge_threshold > split_threshold`` hysteresis,
recompute-after-split, and the budget-conditioned candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.encoder_contract import ComputeBudget, EncoderOutput, MeasurementFrame
from .config import N01Config
from .model import encode_evidence_statistics, statistics_channels
from .state import N01State
from .state import init_state as init_state_fn
from .state import push as push_fn
from .state import reset as reset_fn
from .support import SupportDomainIntegrator

__all__ = [
    "Sequence1DBatch",
    "HeteroscedasticSequence1DTask",
    "N01Sequence1DEncoder",
    "CONV_TYPES_1D",
]


@dataclass
class Sequence1DBatch:
    """One causal window of a one-dimensional heteroscedastic measurement."""

    values: torch.Tensor  # [B, T, L] calibrated measurement
    valid: torch.Tensor  # [B, T, L]
    variance: torch.Tensor  # [B, T, L] declared per-sample variance
    bad: torch.Tensor  # [B, T, L]
    saturated: torch.Tensor  # [B, T, L]
    timestamps: torch.Tensor  # [B, T]
    new_measurement: torch.Tensor  # [B, T] bool
    edge_target: torch.Tensor  # [B, T, L] narrow-gap truth (supervision only)
    target_position: torch.Tensor  # [B, T] normalised position in [0, 1]
    target_visible: torch.Tensor  # [B, T] bool
    meta: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.values.shape[2])


class HeteroscedasticSequence1DTask:
    """Weak narrow gap in a one-dimensional sequence with per-sample variance.

    The generator emits a linear measurement directly (no image is formed), a
    narrow gap that may move along the sequence, Poisson-Gaussian noise plus dead
    and saturated samples, and a per-sample variance that differs by a factor
    between the low- and high-noise halves of the sequence.  The geometry truth is
    for supervision and evaluation only.
    """

    SNR_SETTINGS = {
        "low": (2.0e-2, 0.012, 0.05),
        "medium": (2.0e-3, 0.005, 0.10),
        "high": (2.0e-4, 0.002, 0.20),
    }

    def __init__(
        self,
        config: N01Config,
        *,
        length: int = 64,
        snr: str = "low",
        motion: str = "still",
        seed: int = 0,
        heteroscedastic_ratio: float = 4.0,
        dead_rate: float = 0.01,
        saturation_rate: float = 0.0,
    ):
        if snr not in self.SNR_SETTINGS:
            raise ValueError(f"unknown snr {snr!r}")
        if motion not in {"still", "slow", "fast"}:
            raise ValueError(f"unknown motion {motion!r}")
        if length < 8:
            raise ValueError("the 1-D sequence must be at least 8 samples long")
        self.config = config
        self.length = int(length)
        self.snr = snr
        self.motion = motion
        self.seed = int(seed)
        self.heteroscedastic_ratio = float(heteroscedastic_ratio)
        self.dead_rate = float(dead_rate)
        self.saturation_rate = float(saturation_rate)

    def spec(self) -> dict[str, Any]:
        photon, read, contrast = self.SNR_SETTINGS[self.snr]
        return {
            "task": "synthetic_1d_heteroscedastic_narrow_gap",
            "length": self.length,
            "snr": self.snr,
            "motion": self.motion,
            "seed": self.seed,
            "heteroscedastic_ratio": self.heteroscedastic_ratio,
            "dead_rate": self.dead_rate,
            "saturation_rate": self.saturation_rate,
            "photon_scale": photon,
            "read_std": read,
            "nominal_contrast": contrast,
            "signal_kind": "synthetic_1d_measurement",
        }

    def sample(self, batch_size: int, frames: int, *, seed: int | None = None) -> Sequence1DBatch:
        generator = torch.Generator().manual_seed(self.seed if seed is None else int(seed))
        photon, read_std, contrast = self.SNR_SETTINGS[self.snr]
        span = float({"still": 0.0, "slow": 0.08, "fast": 0.30}[self.motion])
        positions = torch.zeros((batch_size, frames))
        base = torch.empty(batch_size).uniform_(0.2, 0.8, generator=generator)
        direction = torch.where(
            torch.rand(batch_size, generator=generator) > 0.5,
            torch.ones(batch_size),
            -torch.ones(batch_size),
        )
        for step in range(frames):
            progress = (frames - 1 - step) / max(frames - 1, 1)
            positions[:, step] = (base + direction * span * progress).clamp(0.05, 0.95)
        gap_width = torch.empty(batch_size).uniform_(1.0, 3.0, generator=generator)
        columns = torch.arange(self.length, dtype=torch.float32)[None, None, :]
        centre = positions[:, :, None] * (self.length - 1)
        spread = (gap_width[:, None, None] * 0.5) ** 2
        clean = torch.exp(-((columns - centre) ** 2) / (2.0 * spread))
        clean = clean * contrast
        edge_target = clean / max(contrast, 1e-9)

        signal = clean.clamp(min=0.0)
        counts = torch.poisson(signal / max(photon, 1e-9), generator=generator) * photon
        noise = torch.randn(signal.shape, generator=generator) * read_std
        values = (counts + noise).clamp(min=0.0, max=1.0)
        variance = photon * signal + read_std**2
        # Explicit heteroscedasticity: the second half of the sequence is noisier,
        # so a single global noise scale cannot describe it.
        boost = torch.ones(self.length)
        boost[self.length // 2 :] = self.heteroscedastic_ratio
        noisy_extra = (
            torch.randn(signal.shape, generator=generator)
            * read_std
            * torch.sqrt((boost - 1.0) * 0.5)
        )
        half = self.length // 2
        values[:, :, half:] = (values[:, :, half:] + noisy_extra[:, :, half:]).clamp(0.0, 1.0)
        variance = variance * boost[None, None, :]
        saturated = torch.zeros_like(values, dtype=torch.bool)
        if self.saturation_rate > 0.0:
            saturated = torch.rand(values.shape, generator=generator) < self.saturation_rate
            values = torch.where(saturated, torch.ones_like(values), values)
            variance = torch.where(saturated, torch.zeros_like(variance), variance)
        bad = torch.rand(values.shape, generator=generator) < self.dead_rate
        values = torch.where(bad, torch.zeros_like(values), values)
        variance = torch.where(bad, torch.ones_like(variance), variance)
        valid = ~(bad | saturated)
        timestamps = (
            torch.arange(frames, dtype=torch.float32)[None].expand(batch_size, frames).contiguous()
        )
        return Sequence1DBatch(
            values=values,
            valid=valid,
            variance=variance,
            bad=bad,
            saturated=saturated,
            timestamps=timestamps,
            new_measurement=torch.ones((batch_size, frames), dtype=torch.bool),
            edge_target=edge_target,
            target_position=positions,
            target_visible=torch.ones((batch_size, frames), dtype=torch.bool),
            meta={"spec": self.spec()},
        )


#: The one-dimensional convolution types this adapter is allowed to use.  The test
#: suite asserts that no two-dimensional convolution is present anywhere.
CONV_TYPES_1D: tuple[type, ...] = (nn.Conv1d,)


def _groups(width: int) -> int:
    for candidate in (16, 8, 4, 2):
        if width % candidate == 0:
            return candidate
    return 1


class Sequence1DStem(nn.Module):
    """Local 1-D stem over the measurement, its declared variance and validity."""

    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(3, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor, valid: torch.Tensor, variance: torch.Tensor):
        # values is [B,1,L]; the three channels are measurement, log-variance and
        # validity, never a synthetic image.
        inputs = torch.cat([values, torch.log1p(variance), valid.float()], dim=1)
        return self.body(inputs)


class Sequence1DEvidenceBlock(nn.Module):
    """Local exchange and retain/decay update along the sequence."""

    def __init__(self, width: int, statistics: int):
        super().__init__()
        self.exchange = nn.Conv1d(width, width, 3, padding=1, groups=_groups(width))
        self.gate = nn.Sequential(nn.Conv1d(width + statistics, width, 1), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.GroupNorm(_groups(width), width)

    def forward(self, features, statistics, graph):
        batch, channels, length = features.shape
        groups = self.exchange.groups
        patches = F.pad(features, (1, 1)).unfold(-1, 3, 1).permute(0, 1, 3, 2)
        patches = patches.reshape(batch, groups, channels // groups, 3, length)
        kernel = self.exchange.weight.reshape(groups, channels // groups, channels // groups, 3)
        exchanged = torch.einsum("goik,bgikl->bgol", kernel, patches * graph[:, None, None])
        exchanged = exchanged.reshape(batch, channels, length) + self.exchange.bias[None, :, None]
        mixed = features + exchanged
        retain = self.gate(torch.cat([mixed, statistics], dim=1))
        proposal = self.update(mixed)
        return F.relu(mixed + self.norm(retain * proposal + (1.0 - retain) * mixed))


class N01Sequence1DEncoder(nn.Module):
    """N01 applied to a one-dimensional heteroscedastic sequence.

    State, support graph, hysteresis, split-and-recompute and the candidate
    statistics all come from the shared N01 modules; only the stem, the evidence
    blocks and the readout are one-dimensional.
    """

    def __init__(self, config: N01Config):
        super().__init__()
        config.validate()
        if config.content_mode != "gate_only":
            raise ValueError("the 1-D adapter currently supports only content_mode='gate_only'")
        if config.source_reliability != "none":
            raise ValueError("the 1-D adapter does not yet support source_reliability")
        if config.channels != 1:
            raise ValueError("the 1-D adapter expects config.channels == 1")
        self.config = config
        self.integrator = SupportDomainIntegrator(config)
        self.stem = Sequence1DStem(config.width)
        self.statistics_channels = statistics_channels(config)
        self.blocks = nn.ModuleList(
            [
                Sequence1DEvidenceBlock(config.width, self.statistics_channels)
                for _ in range(config.num_blocks)
            ]
        )
        self.readout = nn.Conv1d(config.width, config.width, 1)
        self.last_diagnostics: dict[str, Any] = {}
        self.last_statistics: Any = None
        self.last_connectivity: Any = None
        self.register_buffer("evidence_scales", torch.ones(self.statistics_channels))
        self.register_buffer("evidence_calibration_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("evidence_encoding_version", torch.tensor(1, dtype=torch.long))

    # -- interface ----------------------------------------------------------
    def init_state(self, batch_size: int, length: int, device: str = "cpu") -> N01State:
        return init_state_fn(
            self.config, batch_size, {"camera_ids": ["sequence"], "shape": (1, int(length))}, device
        )

    def reset(self, state: N01State, batch_mask) -> N01State:
        return reset_fn(state, batch_mask)

    def step(
        self,
        frame: MeasurementFrame,
        context: Any = None,
        state: N01State | None = None,
        budget: ComputeBudget | None = None,
    ) -> EncoderOutput:
        if state is None:
            raise ValueError("N01Sequence1DEncoder.step requires a state from init_state")
        budget = budget or ComputeBudget(max_history=self.config.max_history)
        previous = state
        shifted = push_fn(previous, frame)
        fresh = (shifted.filled > previous.filled) | (shifted.frame_ids[0] != previous.frame_ids[0])
        if not bool(fresh.any()) or budget.max_updates <= 0:
            if previous.last_representation is None:
                raise ValueError("no prior representation to reuse")
            return EncoderOutput(
                previous.last_representation,
                shifted,
                previous.valid[0].any(1),
                {"reused_representation": True},
                {"updates": 0},
            )
        measured_edges, edge_info = self.integrator.edge_field(frame, shifted)
        plane = shifted.plane_support("sequence")
        measured = shifted.valid[0].any(dim=1)
        edges, support_age, merges, splits = self.integrator.update_edges(
            plane.edges, plane.support_age, measured_edges, measured, plane.open_edges
        )
        mask = fresh[:, None, None, None, None]
        edges = torch.where(mask, edges, plane.edges)
        self.integrator.last_open_edges = torch.where(
            mask, self.integrator.last_open_edges, plane.open_edges
        )
        support_age = torch.where(mask, support_age, plane.support_age)
        merges, splits = merges * fresh, splits * fresh
        # Write every support field back, exactly as the 2-D encoder does: the
        # cumulative merge/split counters and `seen`/`last_seen` are part of the
        # state, not only of this step's diagnostics.
        now = shifted.now[:, None, None].expand_as(plane.last_seen)
        shifted.support["sequence"] = replace(
            plane,
            edges=edges,
            open_edges=self.integrator.last_open_edges,
            seen=plane.seen | measured[:, None, None],
            support_age=support_age,
            last_seen=torch.where(measured, now, plane.last_seen),
            merge_count=plane.merge_count + merges,
            split_count=plane.split_count + splits,
        )
        statistics, connectivity = self.integrator.domain_statistics(
            shifted, torch.where(self.integrator.last_open_edges, edges, torch.zeros_like(edges))
        )
        weights = self._selection_weights(statistics, connectivity)
        features = self._encode(shifted, statistics, connectivity, weights)
        if previous.last_representation is not None:
            features = torch.where(fresh[:, None, None], features, previous.last_representation)
        shifted.last_representation = features
        self.last_statistics = statistics
        self.last_connectivity = connectivity
        support = shifted.support["sequence"]
        diagnostics = {
            **edge_info,
            "mean_age": float(statistics.age.mean().detach()),
            "mean_support_count": float(statistics.support_count.mean().detach()),
            "merge_count": float(support.merge_count.float().mean().detach()),
            "split_count": float(support.split_count.float().mean().detach()),
            "mixture_participation": float(
                torch.where(weights.sum(0) > 0, 1.0 / weights.square().sum(0).clamp_min(1e-9), 0.0)
                .mean()
                .detach()
            ),
        }
        self.last_diagnostics = diagnostics
        cost = {
            "updates": int(budget.max_updates),
            "candidate_domains": int(statistics.candidate_count),
            "history": int(shifted.filled.max().item()),
        }
        return EncoderOutput(
            representation=features,
            state=shifted,
            validity=measured,
            diagnostics=diagnostics,
            cost=cost,
        )

    # -- internals ----------------------------------------------------------
    def _selection_weights(self, statistics, connectivity) -> torch.Tensor:
        """Same analytic cost shape as the 2-D selector, applied to 1-D candidates."""
        config = self.config
        weights = config.cost_weights
        variance = statistics.variance_proxy.clamp_min(0.0)
        scale = variance.mean().detach().clamp_min(1e-6)
        cost = weights["noise"] * (variance / scale) + weights["edge"] * (1.0 - connectivity)
        cost = cost + weights["age"] * (
            statistics.age / statistics.age.mean().detach().clamp_min(1e-6)
        )
        return torch.softmax(-cost / config.selector_temperature, dim=0)

    def _encode(self, state, statistics, connectivity, weights):
        config = self.config
        values = state.values[0]  # [B, C, 1, L]
        valid = state.valid[0]
        variance = state.variance[0]
        sequence = values.mean(dim=1, keepdim=True).reshape(values.shape[0], 1, -1)
        sequence_valid = (
            valid.float().mean(dim=1, keepdim=True).reshape(valid.shape[0], 1, -1) > 0.5
        )
        sequence_variance = variance.mean(dim=1, keepdim=True).reshape(variance.shape[0], 1, -1)
        features = self.stem(sequence, sequence_valid, sequence_variance)

        def mix(stack: torch.Tensor) -> torch.Tensor:
            # [C,B,P,1,L] -> [B,1,L]
            return (weights * stack).sum(dim=0).reshape(stack.shape[1], 1, -1)

        statistics_tensor = torch.cat(
            [
                mix(statistics.mu),
                mix(statistics.variance_proxy),
                mix(statistics.age),
                mix(statistics.support_count),
                mix(statistics.unique_frames),
                mix(statistics.weight_concentration),
            ],
            dim=1,
        )
        if getattr(self, "calibrating", False) and int(self.evidence_calibration_count) < 64:
            estimates = statistics_tensor.detach().abs().mean(dim=(0, 2)).clamp_min(1e-6)
            with torch.no_grad():
                count = int(self.evidence_calibration_count)
                self.evidence_scales.copy_((self.evidence_scales * count + estimates) / (count + 1))
                self.evidence_calibration_count.add_(1)
        statistics_tensor = encode_evidence_statistics(statistics_tensor, self.evidence_scales)
        if statistics_tensor.shape[1] != self.statistics_channels:
            raise ValueError(
                f"1-D statistics block has {statistics_tensor.shape[1]} channels, blocks "
                f"were built for {self.statistics_channels}"
            )
        support = state.plane_support("sequence")
        strength = torch.where(support.open_edges, support.edges, torch.zeros_like(support.edges))
        graph = torch.stack(
            [
                self.integrator._edge_coefficient(strength, 0, dx, 1)[0][:, 0, 0]
                for dx in (-1, 0, 1)
            ],
            1,
        )
        for index in range(config.num_blocks):
            features = self.blocks[index](features, statistics_tensor, graph)
        return self.readout(features)

    def readout_tokens(self, output: EncoderOutput, token_budget: int | None = None):
        """1-D readout: keep the fine-scale positions instead of a global average."""
        representation = output.representation  # [B, W, L]
        batch, width, length = representation.shape
        budget = min(int(token_budget or self.config.readout_tokens), length)
        index = torch.linspace(0, length - 1, budget).round().to(torch.int64)
        tokens = representation[:, :, index].transpose(1, 2).contiguous()
        positions = (index.to(representation.dtype) / max(length - 1, 1)).reshape(1, budget, 1)
        return tokens, positions.expand(batch, budget, 1)

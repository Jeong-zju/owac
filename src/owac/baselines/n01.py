"""Mechanism probes for the N01 evidence-integration study (prompt step 5).

Implements the "对照和反证实验" list: a single-frame RAW CNN (C3 control), a fixed
spatial/temporal average with a fixed window (including the maximum-history full
average as the degenerate condition), a fixed-constant EMA, a variance-driven
Kalman recursion, a motion-compensated integrator that reuses the N01 causal
correspondence, and a same-history ConvGRU following RAFT's update block, plus
the N01 encoder itself and its merge/split and fixed-horizon ablations.

Boundaries: every probe is causal, reads only ``frames``/``valid``/``variance``/
``timestamps`` (never ``edge_target``/``target_xy`` or any other ground truth),
carries no loss and no localisation head, and returns a feature field
``[B, W_feat, H, W]`` for the shared probe head owned by training/evaluation.
Statistics stay in measurement units; learned features are never reported as
confidence.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from owac.frameworks.n01.config import N01Config

__all__ = [
    "ConvGruIntegrator",
    "EmaIntegrator",
    "FixedAverageIntegrator",
    "KalmanIntegrator",
    "MechanismProbe",
    "MotionCompensatedIntegrator",
    "N01FixedHorizonProbe",
    "N01NoMergeProbe",
    "N01Probe",
    "SingleFrameRawCnn",
    "baseline_registry",
    "mechanism_baselines",
    "strongest_pair",
]

_EPS = 1e-8
_CANDIDATE_NAME = "n01"


# --------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------


def _config_value(config: Any, name: str, default: Any) -> Any:
    """Read a config field lazily so the module imports without ``N01Config`` present."""
    if config is None:
        return default
    return getattr(config, name, default)


def _probe_dims(
    config: Any, width: int | None, channels: int | None, max_history: int | None
) -> tuple[int, int, int]:
    resolved_width = _config_value(config, "width", 32) if width is None else width
    resolved_channels = _config_value(config, "channels", 3) if channels is None else channels
    resolved_history = (
        _config_value(config, "max_history", 8) if max_history is None else max_history
    )
    return int(resolved_width), int(resolved_channels), int(resolved_history)


def _validate_sequence(
    frames: Tensor,
    valid: Tensor | None,
    variance: Tensor | None,
    timestamps: Tensor | None,
    channels: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Check the shared ``[B,T,C,H,W]`` measurement contract and normalise dtypes.

    Non-finite or negative variances are not repaired: they are a loader/noise-model
    error and are clamped only away from zero, so the caller can see the failure.
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError("frames must be a torch.Tensor of shape [B,T,C,H,W]")
    if frames.dim() != 5:
        raise ValueError(f"frames must be [B,T,C,H,W], got shape {tuple(frames.shape)}")
    if frames.shape[2] != channels:
        raise ValueError(f"expected {channels} channels, got {frames.shape[2]}")
    if frames.shape[1] == 0:
        raise ValueError("frames must contain at least one timestep")
    frames = frames.to(torch.float32)
    for label, other in (("valid", valid), ("variance", variance)):
        if other is not None and tuple(other.shape) != tuple(frames.shape):
            raise ValueError(
                f"{label} must match frames {tuple(frames.shape)}, got {tuple(other.shape)}"
            )
    valid_mask = (
        torch.ones_like(frames) if valid is None else valid.to(torch.float32).clamp(0.0, 1.0)
    )
    variance_values = (
        torch.ones_like(frames) if variance is None else variance.to(torch.float32).clamp_min(_EPS)
    )
    times: Tensor | None = None
    if timestamps is not None:
        if tuple(timestamps.shape) != tuple(frames.shape[:2]):
            raise ValueError(
                f"timestamps must match [B,T] == {tuple(frames.shape[:2])}, "
                f"got {tuple(timestamps.shape)}"
            )
        times = timestamps.to(torch.float32)
    return frames, valid_mask, variance_values, times


def _supported(field: Tensor, support: Tensor) -> Tensor:
    """Zero unsupported pixels and append exactly one support channel.

    ``support`` may be per channel (the same channel count as ``field``) or already a
    single mask; the appended channel is the per-pixel union of the support, so every
    caller feeds the stem ``C + 1`` channels.
    """
    support = support.to(field.dtype)
    if support.shape[1] != field.shape[1]:
        if support.shape[1] != 1:
            raise ValueError(f"support has {support.shape[1]} channels, field has {field.shape[1]}")
        support = support.expand(-1, field.shape[1], -1, -1)
    measured = (support > 0.5).any(dim=1, keepdim=True).to(field.dtype)
    return torch.cat([field * support, measured], dim=1)


def _norm_groups(width: int) -> int:
    groups = math.gcd(int(width), 8)
    return max(groups, 1)


def _default_alignment(config: Any) -> str:
    """Motion compensation must actually warp, so ``none`` is promoted to ``fixed``."""
    mode = str(_config_value(config, "alignment", "fixed"))
    return "fixed" if mode == "none" else mode


def _config_with_alignment(config: Any, alignment: str) -> Any:
    if str(_config_value(config, "alignment", "")) == alignment:
        return config
    try:
        return replace(config, alignment=alignment)
    except TypeError as exc:  # pragma: no cover - only for non-dataclass configs
        raise ValueError(
            f"cannot force alignment={alignment!r} on config {type(config).__name__}"
        ) from exc


class _MeasurementStem(nn.Module):
    """3x3 conv stack with GroupNorm (never BatchNorm) and GELU.

    GroupNorm keeps one sample's normalisation independent of the other time steps a
    training batch happens to contain, which BatchNorm would not.
    """

    def __init__(self, in_channels: int, width: int, layers: int = 2) -> None:
        super().__init__()
        groups = _norm_groups(width)
        blocks: list[nn.Module] = []
        current = int(in_channels)
        for _ in range(int(layers)):
            blocks.extend(
                [
                    nn.Conv2d(current, width, 3, padding=1, bias=False),
                    nn.GroupNorm(groups, width),
                    nn.GELU(),
                ]
            )
            current = width
        self.blocks = nn.Sequential(*blocks)

    def forward(self, features: Tensor) -> Tensor:
        return self.blocks(features)


class MechanismProbe(nn.Module):
    """Encode a causal ``[B,T,C,H,W]`` measurement window as ``[B,W_feat,H,W]``.

    ``encode`` never returns coordinates: the shared probe head on the
    training/evaluation side performs localisation, so every mechanism is trained
    with the same head and the same budget. Recurrent probes carry their state
    across ``encode`` calls and expose measured statistics in ``statistics()``.
    """

    name: str = ""
    uses_correspondence: bool = False
    needs_history: bool = True

    def __init__(self, *, width: int = 32, channels: int = 3, max_history: int = 8) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        self.width = int(width)
        self.channels = int(channels)
        self.max_history = int(max_history)
        self._steps = 0

    # -- required interface ------------------------------------------------------------

    def input_spec(self) -> dict[str, Any]:
        """Declare frames, channels, causality, correspondence use and honest notes."""
        raise NotImplementedError

    def encode(
        self,
        frames: Tensor,
        valid: Tensor | None,
        variance: Tensor | None,
        timestamps: Tensor | None = None,
    ) -> Tensor:
        """Encode one causal window; only the current and past frames are read."""
        frames, valid_mask, variance_values, times = _validate_sequence(
            frames, valid, variance, timestamps, self.channels
        )
        indices = range(frames.shape[1]) if self.needs_history else [frames.shape[1] - 1]
        feature: Tensor | None = None
        for index in indices:
            feature = self._step(
                frames[:, index],
                valid_mask[:, index],
                variance_values[:, index],
                None if times is None else times[:, index],
            )
            self._steps += 1
        assert feature is not None  # frames has at least one timestep
        self._detach_state()
        return feature

    def reset_state(self) -> None:
        """Drop every carried measurement/feature state; idempotent."""
        self._steps = 0
        self._clear_state()

    def diagnostics(self) -> dict[str, Any]:
        """Measured bookkeeping for this probe (never a fabricated metric)."""
        return {
            "name": self.name,
            "steps": self._steps,
            "parameters": self.parameter_count(),
        }

    # -- helpers -----------------------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        """Mechanism statistics in measurement units, for numeric checks and reports."""
        return {}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _spec(self, frames: int, notes: str, channels: int | None = None) -> dict[str, Any]:
        return {
            "frames": int(frames),
            "channels": self.channels if channels is None else int(channels),
            "causal": True,
            "uses_correspondence": bool(self.uses_correspondence),
            "notes": notes,
        }

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        raise NotImplementedError

    def _clear_state(self) -> None:
        """Subclass hook: forget all carried state."""

    def _detach_state(self) -> None:
        """Subclass hook: truncate the graph at ``encode`` boundaries (TBPTT)."""


# --------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------


class SingleFrameRawCnn(MechanismProbe):
    """C3 control: an ordinary RAW CNN on the current frame only.

    There is no temporal integration, so this probe caps what a strong single-frame
    feature extractor can do with the same readout budget.
    """

    name = "single_frame_raw_cnn"
    needs_history = False

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        stem_layers: int = 2,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        self.stem = _MeasurementStem(probe_channels + 1, probe_width, layers=stem_layers)

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            1,
            "C3 control: only the final (current) frame of the window is read; earlier "
            "frames are skipped, so there is no temporal integration. GroupNorm stem, "
            "no BatchNorm, no correspondence.",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        self._mean = values * valid
        self._support = valid
        return self.stem(_supported(values, valid))

    def statistics(self) -> dict[str, Any]:
        return {"mean": getattr(self, "_mean", None), "support": getattr(self, "_support", None)}

    def _clear_state(self) -> None:
        self._mean = None
        self._support = None


class FixedAverageIntegrator(MechanismProbe):
    """Step-5 control: fixed spatial/temporal averaging over a fixed window.

    ``window=None`` averages the whole ``max_history`` window, i.e. the maximum-window
    full average. That is the degenerate condition the N01 claim has to beat: if the
    adaptive candidate only reproduces this average, its scheduling is unsupported.
    Weights are uniform; the noise variance never reweights evidence here (that is the
    Kalman baseline's mechanism).
    """

    name = "fixed_average"

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        window: int | None = None,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        if window is not None and int(window) <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.window = probe_history if window is None else min(int(window), probe_history)
        self.stem = _MeasurementStem(probe_channels + 1, probe_width)
        self._buffer: deque[tuple[Tensor, Tensor]] = deque(maxlen=self.window)

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.window,
            f"uniform causal average of the last {self.window} frames "
            f"(window={'max_history' if self.window == self.max_history else self.window}); "
            "no adaptation, no motion compensation; `variance` is accepted for interface "
            "parity but does not reweight evidence.",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        self._buffer.append((values, valid))
        stacked_values = torch.stack([item[0] for item in self._buffer])
        stacked_valid = torch.stack([item[1] for item in self._buffer])
        count = stacked_valid.sum(dim=0)
        total = (stacked_values * stacked_valid).sum(dim=0)
        self._count = count
        self._mean = torch.where(count > 0, total / count.clamp_min(_EPS), torch.zeros_like(total))
        return self.stem(_supported(self._mean, (count > 0).to(values.dtype)))

    def statistics(self) -> dict[str, Any]:
        return {"mean": getattr(self, "_mean", None), "count": getattr(self, "_count", None)}

    def _clear_state(self) -> None:
        self._buffer.clear()
        self._mean = None
        self._count = None

    def _detach_state(self) -> None:
        for index in range(len(self._buffer)):
            values, valid = self._buffer[index]
            self._buffer[index] = (values.detach(), valid.detach())

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {"window": self.window, "degenerate_max_window": self.window == self.max_history}
        )
        return info


class EmaIntegrator(MechanismProbe):
    """Step-5 control: one fixed time constant, updated by recursion.

    ``m_t = (d*w*m + (1-d)*valid*x) / (d*w + (1-d)*valid)``: the decay never adapts to
    noise, motion or budget, which is exactly the difference to the N01 schedule.
    """

    name = "ema"

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        decay: float = 0.9,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        self.stem = _MeasurementStem(probe_channels + 1, probe_width)
        self._mean: Tensor | None = None
        self._weight: Tensor | None = None

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            f"fixed time constant decay={self.decay}; recursive per frame, state persists "
            "across encode calls until reset_state(); no adaptation to noise, motion or "
            "budget; `variance` is accepted for interface parity but unused.",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        if self._mean is None or self._weight is None:
            self._mean = torch.zeros_like(values)
            self._weight = torch.zeros_like(valid)
        new_weight = self.decay * self._weight + (1.0 - self.decay) * valid
        numerator = self.decay * self._weight * self._mean + (1.0 - self.decay) * valid * values
        updated = numerator / new_weight.clamp_min(_EPS)
        self._mean = torch.where(new_weight > 0, updated, self._mean)
        self._weight = new_weight
        support = (self._weight > 0).to(values.dtype)
        return self.stem(_supported(self._mean, support))

    def statistics(self) -> dict[str, Any]:
        return {"mean": self._mean, "weight": self._weight}

    def _clear_state(self) -> None:
        self._mean = None
        self._weight = None

    def _detach_state(self) -> None:
        if self._mean is not None:
            self._mean = self._mean.detach()
        if self._weight is not None:
            self._weight = self._weight.detach()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info["decay"] = self.decay
        return info


class KalmanIntegrator(MechanismProbe):
    """Step-5 control: recursive estimate updated by the measured noise variance.

    Explicit state ``mean``/``var`` per pixel and channel. The first valid observation
    initialises the state; afterwards ``K = V/(V+v)``, ``mean += K*(x-mean)``,
    ``V = (1-K)*V``. No process noise and no decay are added, so the gain is exactly the
    frozen formula and no information beyond the supplied variance enters.
    """

    name = "kalman"

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        eps: float = _EPS,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        self.eps = float(eps)
        self.stem = _MeasurementStem(probe_channels + 1, probe_width)
        self._mean: Tensor | None = None
        self._var: Tensor | None = None
        self._observed: Tensor | None = None

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            "recursive noise-weighted estimate, K = V/(V+v) with the per-pixel supplied "
            "variance; explicit mean/var state, no process noise, no decay, no motion; "
            "state persists across encode calls until reset_state().",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        if self._mean is None or self._var is None or self._observed is None:
            self._mean = torch.zeros_like(values)
            self._var = torch.zeros_like(values)
            self._observed = torch.zeros_like(values, dtype=torch.bool)
        observed = valid > 0
        first = observed & ~self._observed
        update = observed & self._observed
        gain = self._var / (self._var + variance + self.eps)
        candidate_mean = self._mean + gain * (values - self._mean)
        candidate_var = (1.0 - gain) * self._var
        self._mean = torch.where(first, values, torch.where(update, candidate_mean, self._mean))
        self._var = torch.where(first, variance, torch.where(update, candidate_var, self._var))
        self._observed = self._observed | observed
        return self.stem(_supported(self._mean, self._observed.to(values.dtype)))

    def statistics(self) -> dict[str, Any]:
        return {"mean": self._mean, "var": self._var, "observed": self._observed}

    def _clear_state(self) -> None:
        self._mean = None
        self._var = None
        self._observed = None

    def _detach_state(self) -> None:
        if self._mean is not None:
            self._mean = self._mean.detach()
        if self._var is not None:
            self._var = self._var.detach()
        if self._observed is not None:
            self._observed = self._observed.detach()


class MotionCompensatedIntegrator(MechanismProbe):
    """Step-5 control: the N01 correspondence plus a compensated uniform average.

    One correspondence estimate per new frame warps the accumulator (mean, variance
    proxy and support count) into the current frame's coordinates; the new measurement
    then enters with weight 1, exactly as in ``FixedAverageIntegrator``. Keeping the
    weights uniform makes the contrast to the fixed average single-variable: only
    motion alignment differs. There is no support-domain merge/split and no selector.

    The correspondence is the shared ``CausalCorrespondence`` (same module and data as
    N01). ``align`` is called once for the accumulated statistics and once for the
    accumulated count; both are integer/subsample warps of the same field.
    """

    name = "motion_compensated"
    uses_correspondence = True

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        correspondence: nn.Module | None = None,
        alignment: str | None = None,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        self._config = config
        self.alignment = _default_alignment(config) if alignment is None else str(alignment)
        self._correspondence = correspondence
        self.stem = _MeasurementStem(probe_channels + 1, probe_width)
        self._mean: Tensor | None = None
        self._var: Tensor | None = None
        self._count: Tensor | None = None
        self._last_values: Tensor | None = None
        self._last_valid: Tensor | None = None
        self.correspondence_calls = 0

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            f"uniform causal average ({self.max_history}-frame budget) with integer/subpixel "
            f"motion compensation using the shared CausalCorrespondence "
            f"(alignment={self.alignment}, so the same learnable residual gate as the "
            "candidate); one correspondence estimate per new frame; no support-domain "
            "merge/split, no selector, no budget conditioning.",
        )

    def _resolve_correspondence(self) -> Any:
        if self._correspondence is not None:
            return self._correspondence
        if self._config is None:
            raise RuntimeError(
                "MotionCompensatedIntegrator needs an N01Config for CausalCorrespondence "
                "or an explicit `correspondence` module"
            )
        from owac.frameworks.n01.motion import CausalCorrespondence

        self._correspondence = CausalCorrespondence(
            _config_with_alignment(self._config, self.alignment)
        )
        return self._correspondence

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        if self._mean is None or self._var is None or self._count is None:
            self._mean = values
            self._var = variance
            self._count = valid
        else:
            correspondence = self._resolve_correspondence()
            support = self._count > 0
            # ``field`` moves ``other`` (the previous observation) into ``reference``
            # (the current frame); the same field therefore warps the accumulator.
            field = correspondence(values, self._last_values, valid > 0, self._last_valid > 0)
            self.correspondence_calls += 1
            aligned_mean, aligned_var, _ = correspondence.align(
                self._mean, self._var, support, field
            )
            aligned_count = correspondence.align(self._count, self._var, support, field)[0]
            aligned_count = aligned_count.clamp_min(0.0)
            total = aligned_count + valid
            numerator = aligned_count * aligned_mean + valid * values
            denominator = total.clamp_min(_EPS)
            self._mean = torch.where(total > 0, numerator / denominator, aligned_mean)
            # Uniform weights give V = sum_i v_i / N^2; the recursion keeps that form and
            # is reported as a variance proxy, not as an exact posterior.
            self._var = torch.where(
                total > 0,
                (aligned_count**2 * aligned_var + valid * variance) / denominator**2,
                aligned_var,
            )
            self._count = total
        self._last_values = values
        self._last_valid = valid
        support = (self._count > 0).to(values.dtype)
        return self.stem(_supported(self._mean, support))

    def statistics(self) -> dict[str, Any]:
        return {
            "mean": self._mean,
            "variance_proxy": self._var,
            "count": self._count,
        }

    def _clear_state(self) -> None:
        self._mean = None
        self._var = None
        self._count = None
        self._last_values = None
        self._last_valid = None
        self.correspondence_calls = 0

    def _detach_state(self) -> None:
        for attribute in ("_mean", "_var", "_count", "_last_values", "_last_valid"):
            value = getattr(self, attribute)
            if value is not None:
                setattr(self, attribute, value.detach())

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "alignment": self.alignment,
                "correspondence_calls": self.correspondence_calls,
                "count_max": float(self._count.max()) if self._count is not None else 0.0,
            }
        )
        return info


class ConvGruIntegrator(MechanismProbe):
    """Step-5 control: a convolutional GRU over the same history length.

    Organisation follows RAFT ``core/update.py`` (princeton-vl/RAFT, BSD-3-Clause):
    per-frame encoder features, ConvGRU gates ``z``/``r`` with a tanh candidate and
    ``h = (1-z)*h + z*q``, then a small head on the hidden state. The correlation
    lookup of RAFT's iterative flow refinement is replaced by the measurement frame,
    because this baseline does not estimate flow.
    """

    name = "conv_gru"

    def __init__(
        self,
        config: N01Config | None = None,
        *,
        width: int | None = None,
        channels: int | None = None,
        max_history: int | None = None,
        hidden: int | None = None,
    ) -> None:
        probe_width, probe_channels, probe_history = _probe_dims(
            config, width, channels, max_history
        )
        super().__init__(width=probe_width, channels=probe_channels, max_history=probe_history)
        self.hidden = probe_width if hidden is None else int(hidden)
        groups = _norm_groups(self.hidden)
        self.input_conv = nn.Sequential(
            nn.Conv2d(probe_channels + 1, self.hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, self.hidden),
            nn.GELU(),
        )
        self.convz = nn.Conv2d(2 * self.hidden, self.hidden, 3, padding=1)
        self.convr = nn.Conv2d(2 * self.hidden, self.hidden, 3, padding=1)
        self.convq = nn.Conv2d(2 * self.hidden, self.hidden, 3, padding=1)
        self.head = _MeasurementStem(self.hidden, probe_width)
        self._hidden: Tensor | None = None

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            f"same {self.max_history}-frame history as N01, recurrent hidden state of width "
            f"{self.hidden} (RAFT-style ConvGRU); no correspondence, no support domains, no "
            "budget conditioning; state persists across encode calls until reset_state().",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        features = self.input_conv(_supported(values, valid))
        hidden = torch.zeros_like(features) if self._hidden is None else self._hidden
        gates = torch.cat([hidden, features], dim=1)
        update_gate = torch.sigmoid(self.convz(gates))
        reset_gate = torch.sigmoid(self.convr(gates))
        candidate = torch.tanh(self.convq(torch.cat([reset_gate * hidden, features], dim=1)))
        self._hidden = (1.0 - update_gate) * hidden + update_gate * candidate
        return self.head(self._hidden)

    def statistics(self) -> dict[str, Any]:
        return {"hidden": self._hidden}

    def _clear_state(self) -> None:
        self._hidden = None

    def _detach_state(self) -> None:
        if self._hidden is not None:
            self._hidden = self._hidden.detach()


# --------------------------------------------------------------------------------------
# candidate and candidate ablations
# --------------------------------------------------------------------------------------


def _require_n01() -> tuple[Any, Any, Any]:
    """Import the shared contract and the N01 encoder at call time.

    They are imported lazily so this module stays importable (and the six mechanism
    baselines stay testable) while the candidate modules are still being written.
    """
    from owac.core.encoder_contract import ComputeBudget, MeasurementFrame
    from owac.frameworks.n01.model import N01Encoder

    return N01Encoder, MeasurementFrame, ComputeBudget


class N01Probe(MechanismProbe):
    """Candidate probe: the N01 encoder stepped frame by frame.

    Wraps ``owac.frameworks.n01.model.N01Encoder``: every window frame becomes one
    causal ``MeasurementFrame`` carrying the values, the validity mask and the supplied
    per-pixel variance, and the returned ``EncoderOutput.representation`` field is the
    localisation feature. Token readout stays on the training/evaluation side, exactly as
    for the baselines. Diagnostics sum the measured merge/split transition counts and
    report the temporal depth the encoder actually applied.
    """

    name = "n01"

    def __init__(
        self,
        config: N01Config,
        *,
        camera_id: str = "camera",
        max_updates: int | None = None,
    ) -> None:
        if config is None:
            raise ValueError("N01Probe requires the candidate N01Config")
        super().__init__(
            width=int(_config_value(config, "width", 32)),
            channels=int(_config_value(config, "channels", 3)),
            max_history=int(_config_value(config, "max_history", 8)),
        )
        self.config = config
        self.camera_id = str(camera_id)
        default_updates = int(_config_value(config, "num_blocks", 1))
        self.max_updates = int(default_updates if max_updates is None else max_updates)
        if self.max_updates <= 0:
            raise ValueError(f"max_updates must be positive, got {self.max_updates}")
        encoder_class, _, _ = _require_n01()
        self.encoder = encoder_class(config)
        self.encoder_state: Any | None = None
        self._totals: dict[str, float] = {}
        self._last: dict[str, Any] = {}
        self._cost: dict[str, Any] = {}
        self._depth = 0
        self._depth_levels = 0
        self.uses_correspondence = str(_config_value(config, "alignment", "none")) != "none"

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            "N01 candidate: support-domain reorganisable multi-timescale integration "
            "(hysteretic merge/split support graph, analytic domain selector, "
            f"time_depths={tuple(self.config.time_depths)}, "
            f"budget_conditioning={bool(self.config.budget_conditioning)}, "
            f"max_updates={self.max_updates}); same causal correspondence and history "
            "length as the motion-compensated baseline.",
        )

    def _step(
        self, values: Tensor, valid: Tensor, variance: Tensor, timestamp: Tensor | None
    ) -> Tensor:
        _, frame_class, budget_class = _require_n01()
        if self.encoder_state is None:
            camera_spec = {
                "camera_ids": [self.camera_id],
                "shape": (int(values.shape[-2]), int(values.shape[-1])),
            }
            self.encoder_state = self.encoder.init_state(
                int(values.shape[0]), camera_spec, str(values.device)
            )
        frame = frame_class(
            values=values,
            valid_mask=valid > 0.5,
            camera_id=self.camera_id,
            timestamp=timestamp,
            time_unit="frame",
            metadata={"variance": variance},
        )
        budget = budget_class(max_updates=self.max_updates, max_history=self.max_history)
        output = self.encoder.step(frame, None, self.encoder_state, budget)
        self.encoder_state = output.state
        self._record(output.diagnostics, output.cost)
        return output.representation

    def _clear_state(self) -> None:
        self.encoder_state = None
        self._totals = {}
        self._last = {}
        self._cost = {}
        self._depth = 0
        self._depth_levels = 0

    def _detach_state(self) -> None:
        if self.encoder_state is not None:
            self.encoder_state = self.encoder.detach_state(self.encoder_state)

    def _record(self, diagnostics: dict[str, Any], cost: dict[str, Any]) -> None:
        """Aggregate one step's encoder diagnostics into the probe's measured record."""
        for key in ("merge_count", "split_count"):
            if key in diagnostics:
                self._totals[key] = float(diagnostics[key])
        depths = tuple(int(item) for item in _config_value(self.config, "time_depths", ()))
        for key in ("edge_mean", "mean_support_width", "mean_variance_proxy", "support_age_mean"):
            if key in diagnostics:
                self._last[key] = float(diagnostics[key])
        for key in ("selected_depth_mean", "candidates_considered", "selection_entropy"):
            if key in diagnostics:
                self._last[key] = float(diagnostics[key])
        if "selected_depth" in diagnostics and depths:
            # ``selected_depth`` counts the temporal levels the budget plan kept;
            # ``depth`` is the longest temporal length in frames available after planning.
            levels = max(int(diagnostics["selected_depth"]), 1)
            index = min(levels, len(depths)) - 1
            self._depth = max(self._depth, depths[index])
            self._depth_levels = max(self._depth_levels, levels)
        if "history" in cost:
            self._last["history"] = int(cost["history"])
        if "state_bytes" in cost:
            self._last["state_bytes"] = int(cost["state_bytes"])
        self._cost = dict(cost)

    def statistics(self) -> dict[str, Any]:
        return {"encoder_state": self.encoder_state}

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "merge_count": float(self._totals.get("merge_count", 0.0)),
                "split_count": float(self._totals.get("split_count", 0.0)),
                "depth": int(self._depth),
                "depth_levels": int(self._depth_levels),
                "candidates_considered": self._last.get("candidates_considered"),
                "selected_depth_mean": self._last.get("selected_depth_mean"),
                "edge_mean": self._last.get("edge_mean"),
                "mean_support_width": self._last.get("mean_support_width"),
                "history": self._last.get("history"),
                "state_bytes": self._last.get("state_bytes"),
                "cost": dict(self._cost),
                "time_depths": tuple(int(item) for item in self.config.time_depths),
                "budget_conditioning": bool(self.config.budget_conditioning),
            }
        )
        return info


def _no_merge_config(config: N01Config) -> N01Config:
    """Force every one-hop edge open: no hysteresis barrier and no split condition.

    ``merge_threshold=0`` opens every edge on the first measured update (gate values
    are non-negative) and ``split_threshold=-1`` makes the refutation condition
    unreachable, because validation only requires ``split_threshold <
    merge_threshold``. Both the hysteresis transitions and the domain split condition
    therefore disappear.
    """
    return replace(
        config,
        merge_threshold=0.0,
        split_threshold=-1.0,
        selection="analytic",
        selection_use_learned=False,
    )


def _fixed_horizon_config(config: N01Config) -> N01Config:
    """Fix the temporal depth at the longest configured level and drop conditioning."""
    depths = tuple(int(item) for item in _config_value(config, "time_depths", (1,))) or (1,)
    return replace(config, time_depths=(max(depths),), budget_conditioning=False)


class N01NoMergeProbe(N01Probe):
    """Candidate ablation: every support edge stays open, no hysteresis and no split.

    ``merge_threshold=0`` removes the barrier (every gate value opens its edge) and
    ``split_threshold=-1`` makes the split condition unreachable, so every neighbour
    inside the radius keeps its support weight. The analytic selector is pinned so the contrast
    isolates the edge rule; everything else is the candidate encoder.
    """

    name = "n01_no_merge"

    def __init__(self, config: N01Config, **kwargs: Any) -> None:
        super().__init__(_no_merge_config(config), **kwargs)

    def input_spec(self) -> dict[str, Any]:
        return self._spec(
            self.max_history,
            "ablation: merge_threshold=0 with split_threshold=-1 keeps all one-hop "
            "edges permanently open (no hysteresis, no split, no occlusion release) under "
            "the analytic selector; otherwise the same encoder, correspondence and budget "
            "as n01.",
        )


class N01FixedHorizonProbe(N01Probe):
    """Candidate ablation: temporal depth fixed at the longest level, no conditioning.

    ``time_depths`` collapses to the single maximum level and ``budget_conditioning`` is
    disabled, so the representation can never shorten its history under a budget.
    """

    name = "n01_fixed_horizon"

    def __init__(self, config: N01Config, **kwargs: Any) -> None:
        super().__init__(_fixed_horizon_config(config), **kwargs)

    def input_spec(self) -> dict[str, Any]:
        longest = max(int(item) for item in self.config.time_depths)
        return self._spec(
            self.max_history,
            f"ablation: time depth fixed at {longest} frames and budget conditioning "
            "disabled; otherwise the same encoder, correspondence and history length as n01.",
        )


# --------------------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------------------


def baseline_registry() -> dict[str, Callable[[N01Config], MechanismProbe]]:
    """Return every N01-configurable mechanism probe, keyed by its short name.

    The candidate itself is present under ``n01`` so one table can materialise every row
    of the mechanism matrix (including the two candidate ablations). ``mechanism_baselines``
    is the baseline-only subset required by prompt step 5.
    """
    return {
        SingleFrameRawCnn.name: SingleFrameRawCnn,
        FixedAverageIntegrator.name: FixedAverageIntegrator,
        EmaIntegrator.name: EmaIntegrator,
        KalmanIntegrator.name: KalmanIntegrator,
        MotionCompensatedIntegrator.name: MotionCompensatedIntegrator,
        ConvGruIntegrator.name: ConvGruIntegrator,
        N01Probe.name: N01Probe,
        N01NoMergeProbe.name: N01NoMergeProbe,
        N01FixedHorizonProbe.name: N01FixedHorizonProbe,
    }


def mechanism_baselines(config: N01Config) -> dict[str, MechanismProbe]:
    """Materialise the prompt's "较全" baseline set for one shared configuration.

    That set is: single-frame RAW CNN, fixed average, EMA, Kalman, motion-compensated
    integration with the same correspondence, same-history ConvGRU, and the candidate
    with merge/split or with fixed horizon removed. The candidate itself is not a
    baseline and is therefore not in this table.
    """
    registry = baseline_registry()
    names = [name for name in registry if name != _CANDIDATE_NAME]
    return {name: registry[name](config) for name in names}


def strongest_pair(metrics: Mapping[str, Any]) -> list[str]:
    """Return the two strongest baselines by smallest ``position_error``.

    ``metrics`` maps a probe name to a mapping (or object) carrying ``position_error``.
    Entries without a finite ``position_error`` are ignored, ties break by name, and the
    N01 candidate itself is excluded because it is the method under test, not a baseline.
    """
    scored: list[tuple[float, str]] = []
    for name, entry in metrics.items():
        if str(name) == _CANDIDATE_NAME:
            continue
        if isinstance(entry, Mapping):
            value = entry.get("position_error")
        else:
            value = getattr(entry, "position_error", None)
        if value is None:
            continue
        score = float(value)
        if not math.isfinite(score):
            continue
        scored.append((score, str(name)))
    if len(scored) < 2:
        raise ValueError(
            "strongest_pair needs at least two baselines with a finite position_error, "
            f"got {len(scored)} from {sorted(map(str, metrics))}"
        )
    scored.sort()
    return [name for _, name in scored[:2]]

"""Synthetic weak-edge / narrow-gap localisation task for N01.

The generator produces a *linear* measurement field directly (no RGB image is
formed anywhere), samples it through a declared CFA grid or a three-channel
measurement, adds Poisson-Gaussian noise plus dead pixels, saturation and
motion, and keeps the geometry truth separate.  The geometry truth is used only
for supervision and evaluation; ``values`` is the only thing handed to the
encoder, and ``valid``/``bad``/``saturated`` masks are declared explicitly.

Training samples are generated online from a seeded generator; the validation
split uses an independent seed and combinations of speed/contrast that the
training conditions do not contain, so a validation number is never a re-draw
of a training sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import N01Config

__all__ = ["SequenceBatch", "SyntheticWeakEdgeTask", "SceneSpec", "TASK_CONDITIONS"]

# Low / medium / high SNR x still / slow / fast motion, plus event conditions.
TASK_CONDITIONS: tuple[tuple[str, str], ...] = tuple(
    (snr, motion) for snr in ("low", "medium", "high") for motion in ("still", "slow", "fast")
) + (
    ("low", "sudden_appear"),
    ("low", "direction_reversal"),
    ("low", "occlusion_release"),
    ("medium", "crossing"),
)


@dataclass
class SceneSpec:
    """Geometry truth of one generated sequence (supervision/evaluation only)."""

    target_xy: torch.Tensor  # [B,T,2] normalised (row, col) in [0, 1]
    target_visible: torch.Tensor  # [B,T] bool
    region_scale: float
    contrast: float
    snr: str
    motion: str
    tolerance: str
    seed: int


@dataclass
class SequenceBatch:
    """One causal measurement window plus its supervision-only geometry truth."""

    values: torch.Tensor  # [B,T,C,H,W] calibrated measurement in [0, 1]
    valid: torch.Tensor  # [B,T,C,H,W] usable sample
    variance: torch.Tensor  # [B,T,C,H,W] declared noise variance, same unit
    saturated: torch.Tensor  # [B,T,C,H,W]
    bad: torch.Tensor  # [B,T,C,H,W]
    timestamps: torch.Tensor  # [B,T]
    new_measurement: torch.Tensor  # [B,T] bool
    edge_target: torch.Tensor  # [B,T,H,W] sub-pixel edge strength truth
    target_xy: torch.Tensor  # [B,T,2] normalised (row, col)
    target_visible: torch.Tensor  # [B,T] bool
    scene: SceneSpec
    meta: dict = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        return int(self.values.shape[1])

    def to(self, device) -> SequenceBatch:
        move = lambda tensor: tensor.to(device)  # noqa: E731
        return SequenceBatch(
            values=move(self.values),
            valid=move(self.valid),
            variance=move(self.variance),
            saturated=move(self.saturated),
            bad=move(self.bad),
            timestamps=move(self.timestamps),
            new_measurement=move(self.new_measurement),
            edge_target=move(self.edge_target),
            target_xy=move(self.target_xy),
            target_visible=move(self.target_visible),
            scene=self.scene,
            meta=dict(self.meta),
        )


_SNR_SETTINGS = {
    # (photon scale, read noise std, contrast) for a normalised [0, 1] field.
    "low": (2.0e-4, 0.010, 0.05),
    "medium": (2.0e-3, 0.004, 0.10),
    "high": (2.0e-2, 0.0015, 0.20),
}

_MOTION_SETTINGS = {
    "still": 0.0,
    "slow": 0.35,
    "fast": 1.6,
    "sudden_appear": 0.0,
    "direction_reversal": 0.9,
    "occlusion_release": 0.25,
    "crossing": 1.1,
}

_TOLERANCE_SETTINGS = {"wide": 2.5, "narrow": 0.75}


class SyntheticWeakEdgeTask:
    """Online generator of weak-edge / narrow-gap localisation sequences."""

    def __init__(
        self,
        config: N01Config,
        *,
        image_size: int = 32,
        snr: str = "low",
        motion: str = "still",
        tolerance: str = "wide",
        seed: int = 0,
        augment: str = "none",
        cfa: bool = False,
        dead_pixel_rate: float = 0.002,
        saturation_fraction: float = 0.0,
        noise_scale: float = 1.0,
        device: str = "cpu",
    ):
        if snr not in _SNR_SETTINGS:
            raise ValueError(f"Unknown snr {snr!r}")
        if motion not in _MOTION_SETTINGS:
            raise ValueError(f"Unknown motion {motion!r}")
        if tolerance not in _TOLERANCE_SETTINGS:
            raise ValueError(f"Unknown tolerance {tolerance!r}")
        self.config = config
        self.image_size = int(image_size)
        self.snr = snr
        self.motion = motion
        self.tolerance = tolerance
        self.seed = int(seed)
        self.augment = augment
        self.cfa = bool(cfa)
        self.dead_pixel_rate = float(dead_pixel_rate)
        self.saturation_fraction = float(saturation_fraction)
        self.noise_scale = float(noise_scale)
        self.device = device
        self.condition = (snr, motion)
        self.channels = 4 if self.cfa else config.channels

    # -- specification ------------------------------------------------------
    def spec(self) -> dict:
        photon, read, contrast = _SNR_SETTINGS[self.snr]
        return {
            "task": "synthetic_weak_edge_narrow_gap_localisation",
            "image_size": self.image_size,
            "snr": self.snr,
            "motion": self.motion,
            "motion_amplitude_px": _MOTION_SETTINGS[self.motion],
            "tolerance": self.tolerance,
            "tolerance_px": _TOLERANCE_SETTINGS[self.tolerance],
            "seed": self.seed,
            "augment": self.augment,
            "cfa": self.cfa,
            "cfa_pattern": self.config.cfa_pattern if self.cfa else None,
            "channels": self.channels,
            "photon_scale": photon,
            "read_std": read,
            "nominal_contrast": contrast,
            "noise_scale": self.noise_scale,
            "dead_pixel_rate": self.dead_pixel_rate,
            "saturation_fraction": self.saturation_fraction,
            "signal_kind": "synthetic_sensor_measurement",
            "note": (
                "linear measurement field, explicit CFA sampling and Poisson-Gaussian noise; "
                "the geometry truth is used for supervision and evaluation only"
            ),
        }

    # -- generation ---------------------------------------------------------
    def sample(self, batch_size: int, length: int, *, seed: int | None = None) -> SequenceBatch:
        if batch_size < 1 or length < 1:
            raise ValueError("batch_size and length must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed if seed is None else int(seed))
        size = self.image_size
        device = self.device
        photon, read_std, contrast = _SNR_SETTINGS[self.snr]
        photon = photon * self.noise_scale
        read_std = read_std * math.sqrt(self.noise_scale)
        amplitude = _MOTION_SETTINGS[self.motion]

        # Geometry: one narrow slit of variable width plus a second target that
        # may separate from the first (the "two targets split apart" trajectory).
        target_xy = torch.zeros((batch_size, length, 2))
        second_xy = torch.zeros((batch_size, length, 2))
        visible = torch.ones((batch_size, length), dtype=torch.bool)
        width_px = torch.empty(batch_size).uniform_(0.6, 2.0, generator=generator)
        direction = torch.where(
            torch.rand(batch_size, generator=generator) > 0.5,
            torch.ones(batch_size),
            -torch.ones(batch_size),
        )
        base = torch.empty(batch_size, 2).uniform_(0.3, 0.7, generator=generator)
        separation = torch.empty(batch_size).uniform_(0.05, 0.18, generator=generator)
        for step in range(length):
            phase = 1.0 - step / max(length - 1, 1)
            progress = (length - 1 - step) / max(length - 1, 1)
            if self.motion in ("still", "sudden_appear", "occlusion_release"):
                offset = torch.zeros(batch_size)
            elif self.motion == "direction_reversal":
                offset = (
                    torch.where(torch.tensor(progress > 0.5), direction, -direction)
                    * amplitude
                    * (size / 64.0)
                    * (1.0 - progress)
                )
            else:
                offset = direction * amplitude * (size / 64.0) * progress
            target_xy[:, step, 1] = (base[:, 1] + offset / size * 2.0).clamp(0.08, 0.92)
            target_xy[:, step, 0] = base[:, 0].clamp(0.08, 0.92)
            second_xy[:, step, 1] = (base[:, 1] + offset / size * 2.0).clamp(0.08, 0.92)
            second_xy[:, step, 0] = (base[:, 0] + separation * progress).clamp(0.08, 0.92)
            del phase
        if self.motion == "sudden_appear":
            visible[:, : max(length // 2, 1)] = False
        if self.motion == "occlusion_release":
            visible[:, length // 3 : 2 * length // 3] = False

        clean, edge_truth = self._render(target_xy, second_xy, visible, width_px, contrast, size)
        values, variance, saturated, bad = self._measure(clean, generator, photon, read_std)
        valid = ~(bad | saturated)
        if self.cfa:
            # Only the plane whose phase matches the mosaic site holds a
            # measurement there; the other three planes are missing at that site.
            present = torch.zeros_like(valid)
            for row in range(2):
                for col in range(2):
                    present[:, :, row * 2 + col, row::2, col::2] = True
            valid = valid & present
        # Timestamps are declared per batch member as [B, T]; without a real clock
        # the unit stays "frame" and the index is used unchanged.
        timestamps = (
            torch.arange(length, dtype=torch.float32)[None].expand(batch_size, length).contiguous()
        )
        new_measurement = torch.ones((batch_size, length), dtype=torch.bool)
        scene = SceneSpec(
            target_xy=target_xy,
            target_visible=visible,
            region_scale=float(width_px.mean()),
            contrast=contrast,
            snr=self.snr,
            motion=self.motion,
            tolerance=self.tolerance,
            seed=self.seed if seed is None else int(seed),
        )
        return SequenceBatch(
            values=values.to(device),
            valid=valid.to(device),
            variance=variance.to(device),
            saturated=saturated.to(device),
            bad=bad.to(device),
            timestamps=timestamps.to(device),
            new_measurement=new_measurement.to(device),
            edge_target=edge_truth.to(device),
            target_xy=target_xy.to(device),
            target_visible=visible.to(device),
            scene=scene,
            meta={
                "spec": self.spec(),
                "tolerance_px": _TOLERANCE_SETTINGS[self.tolerance],
                "second_target_xy": second_xy.to(device),
            },
        )

    def _render(self, target_xy, second_xy, visible, width_px, contrast, size):
        """Linear slit fields in the measurement domain, plus the edge truth."""
        device = torch.device("cpu")
        batch, length = target_xy.shape[0], target_xy.shape[1]
        rows = torch.arange(size, dtype=torch.float32)[None, :, None]
        cols = torch.arange(size, dtype=torch.float32)[None, None, :]
        clean = torch.zeros((batch, length, size, size))
        edges = torch.zeros((batch, length, size, size))
        for step in range(length):
            centre_col = target_xy[:, step, 1][:, None, None] * (size - 1)
            second_row = second_xy[:, step, 0][:, None, None] * (size - 1)
            second_col = second_xy[:, step, 1][:, None, None] * (size - 1)
            slit = width_px[:, None, None]
            first = (cols - centre_col).abs() < slit * 0.5
            second = ((rows - second_row).abs() < slit * 0.5) & (
                (cols - second_col).abs() < slit * 0.5
            )
            field = first.float() * contrast + second.float() * contrast
            edge = torch.exp(-((cols - centre_col) ** 2) / (2.0 * (slit * 0.5) ** 2))
            edge = edge + torch.exp(
                -((rows - second_row) ** 2 + (cols - second_col) ** 2) / (2.0 * slit**2)
            )
            gate = visible[:, step][:, None, None].float()
            clean[:, step] = field * gate
            edges[:, step] = edge.clamp(max=1.0) * gate
        del device
        return clean, edges

    def _measure(self, clean, generator, photon, read_std):
        """Poisson-Gaussian measurement in the plane grid, then the CFA layout.

        The photoelectronic model is declared in the same numeric unit as the
        calibrated measurement: ``variance = photon_scale * signal + read_std^2``.
        Dead and saturated sites are flagged, never silently replaced by a value
        that would look like a confident observation.
        """
        batch, length, size, _ = clean.shape
        signal = clean.clamp(min=0.0)
        mean = torch.poisson(signal / max(photon, 1e-9), generator=generator) * photon
        read = torch.randn(signal.shape, generator=generator) * read_std
        noisy = (mean + read).clamp(min=0.0, max=1.0)
        variance = photon * signal + read_std**2
        saturated = torch.zeros_like(noisy, dtype=torch.bool)
        if self.saturation_fraction > 0.0:
            row = torch.rand(batch, length, size, 1, generator=generator)
            saturated = row.expand_as(noisy) < self.saturation_fraction
            noisy = torch.where(saturated, torch.ones_like(noisy), noisy)
            variance = torch.where(saturated, torch.zeros_like(variance), variance)
        bad = torch.rand(batch, length, size, size, generator=generator) < self.dead_pixel_rate
        noisy = torch.where(bad, torch.zeros_like(noisy), noisy)
        variance = torch.where(bad, torch.ones_like(variance), variance)

        target = torch.zeros((batch, length, self.channels, size, size), dtype=torch.float32)
        variance_out = torch.zeros_like(target)
        saturated_out = torch.zeros_like(target, dtype=torch.bool)
        bad_out = torch.zeros_like(target, dtype=torch.bool)
        if self.cfa:
            target = self._mosaic(noisy, target)
            variance_out = self._mosaic(variance, variance_out)
            saturated_out = self._mosaic(saturated, saturated_out)
            bad_out = self._mosaic(bad, bad_out)
            # A site that carries no measurement for this plane is missing, not a
            # measured zero, so the validity mask must say so.
            present = torch.zeros_like(target, dtype=torch.bool)
            for row in range(2):
                for col in range(2):
                    present[:, :, row * 2 + col, row::2, col::2] = True
            variance_out = torch.where(present, variance_out, torch.ones_like(variance_out))
        else:
            target = noisy.unsqueeze(2).expand(-1, -1, self.channels, -1, -1).contiguous()
            variance_out = variance.unsqueeze(2).expand(-1, -1, self.channels, -1, -1).contiguous()
            saturated_out = (
                saturated.unsqueeze(2).expand(-1, -1, self.channels, -1, -1).contiguous()
            )
            bad_out = bad.unsqueeze(2).expand(-1, -1, self.channels, -1, -1).contiguous()
        return target, variance_out, saturated_out, bad_out

    # Colour label of each of the four CFA sites, indexed by plane = row * 2 + col.
    CFA_COLOURS = {
        "RGGB": ("R", "G", "G", "B"),
        "BGGR": ("B", "G", "G", "R"),
        "GRBG": ("G", "R", "B", "G"),
        "GBRG": ("G", "B", "R", "G"),
    }

    def plane_colours(self) -> tuple[str, ...]:
        """Colour label of each measurement plane, in plane order."""
        if not self.cfa:
            return tuple("M" for _ in range(self.channels))
        return self.CFA_COLOURS[self.config.cfa_pattern]

    def _mosaic(self, plane_values, target):
        """One filtered measurement per mosaic site, reordered into four planes.

        Each site of the 2x2 tile carries exactly one filter measurement, so the
        mosaic reorders into four planes with sampling phases (0,0), (0,1), (1,0)
        and (1,1).  The planes are separate measurements at different positions,
        never four co-sited colours, and no value is interpolated: the remaining
        channels of a site stay zero (and are masked as missing, not as a
        measured zero).
        """
        for row in range(2):
            for col in range(2):
                plane = row * 2 + col
                target[:, :, plane, row::2, col::2] = plane_values[:, :, row::2, col::2]
        return target

    def sample_xy(self) -> torch.Tensor:
        """Declared per-plane sampling offset (row, col) of the CFA phase."""
        if not self.cfa:
            return torch.zeros((self.channels, 2), dtype=torch.float32)
        offsets = torch.zeros((self.channels, 2), dtype=torch.float32)
        for row in range(2):
            for col in range(2):
                plane = row * 2 + col
                offsets[plane, 0] = float(row)
                offsets[plane, 1] = float(col)
        return offsets


def localisation_error_px(prediction: torch.Tensor, target_xy: torch.Tensor, size: int) -> float:
    """Mean position error in pixels for a normalised (row, col) prediction."""
    delta = (prediction - target_xy[:, :2]) * size
    return float(delta.norm(dim=-1).mean())


def blur_target(edge_target: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian heatmap target used by the heatmap-cross-entropy localisation loss."""
    if sigma <= 0:
        return edge_target
    radius = max(int(2 * sigma), 1)
    coords = torch.arange(-radius, radius + 1, dtype=edge_target.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    batch, length, height, width = edge_target.shape
    flat = edge_target.reshape(batch * length, 1, height, width)
    flat = F.conv2d(flat, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    flat = F.conv2d(flat, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return flat.reshape(batch, length, height, width)

"""Independent native-CFA fixtures for the N01 content-path factorial experiment.

These are linear spectral measurements sampled at four physical RGGB offsets,
not mosaicked RGB images or copies of one noisy channel. Geometry is supervision
only, except the explicitly labelled exact-translation positive control.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

CONDITIONS = (
    "static_weak",
    "translation_oracle_alignment",
    "occlusion_release",
    "local_change_stale_history",
)
CFA_OFFSETS = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass
class ContentBatch:
    values: torch.Tensor
    variance: torch.Tensor
    valid: torch.Tensor
    target_xy: torch.Tensor  # packed-grid cells, never passed to ordinary encoder
    visible: torch.Tensor
    translations: torch.Tensor  # known fixture camera shift; oracle control only
    condition: str
    event_frame: int
    seed: int

    def input_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(repr(CFA_OFFSETS).encode())
        for tensor in (self.values, self.variance, self.valid):
            digest.update(tensor.contiguous().cpu().numpy().tobytes())
        if self.condition == "translation_oracle_alignment":
            digest.update(self.translations.contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    def fixture_hash(self) -> str:
        digest = hashlib.sha256(self.input_hash().encode())
        digest.update(f"{self.condition}:{self.seed}:{self.event_frame}".encode())
        for tensor in (self.target_xy, self.visible, self.translations):
            digest.update(tensor.contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()


def split_seed(seed: int, split: str, index: int) -> int:
    """Disjoint bounded seed ranges, independent of method and execution order."""
    offsets = {"calibration": 0, "train": 1, "validation": 2, "test": 3}
    if split not in offsets or not 0 <= seed < 10000 or not 0 <= index < 100000:
        raise ValueError("invalid split, seed or index")
    return offsets[split] * 1_000_000_000 + seed * 100000 + index


def make_batch(
    condition: str,
    *,
    seed: int,
    batch_size: int = 2,
    size: int = 16,
    length: int = 24,
    event_frame: int = 12,
    contrast: float = 0.08,
    noise_sigma: float = 0.08,
) -> ContentBatch:
    """Causal per-frame scenes; future event choices never modify their prefixes.

    An occluder is a measured opaque surface, not an invalid-pixel mask. The
    localized change moves the target to a new location; old measurements were
    correct when acquired but become misleading afterwards. Scene noise remains
    independent of geometry and conditions share the same seeded draws.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if size < 12 or not 16 <= length < 10000 or not 8 <= event_frame <= length - 6:
        raise ValueError("need size>=12 and at least six measured post-event frames")
    if noise_sigma < 0 or contrast <= 0 or batch_size < 1:
        raise ValueError("invalid sensor parameters")
    generator = torch.Generator().manual_seed(seed)
    # Stateless generation per frame makes longer sequences have identical
    # prefixes; varying T cannot change the random stream of another member.
    centres = 3.0 + torch.rand(batch_size, 2, generator=generator) * (size / 2 - 5.0)
    flip = torch.rand(batch_size, 2, generator=generator) > 0.5
    centres = torch.where(flip, (size - 1) - centres, centres)
    next_centres = (size - 1) - centres
    target = centres[:, None].expand(-1, length, -1).clone()
    translation = torch.zeros(batch_size, length, 2)
    if condition == "translation_oracle_alignment":
        # Integer packed-cell translations, two physical CFA pixels per cell.
        pattern = torch.tensor([0, 1, 2, 1, 0, -1, -2, -1], dtype=torch.float32)
        for frame in range(8, length):
            translation[:, frame, 1] = pattern[(frame - 7) % len(pattern)]
        target += translation
    if condition == "local_change_stale_history":
        target[:, event_frame:] = next_centres[:, None]
    visible = torch.ones(batch_size, length, dtype=torch.bool)
    if condition == "occlusion_release":
        visible[:, event_frame - 4 : event_frame] = False
    offsets = torch.tensor(CFA_OFFSETS, dtype=torch.float32)
    rows = torch.arange(size)[None, None, :, None] + offsets[None, :, 0, None, None] / 2
    cols = torch.arange(size)[None, None, None, :] + offsets[None, :, 1, None, None] / 2
    # G1/G2 have the same spectral response but distinct positions/noise draws.
    target_spectrum = torch.tensor([1.0, 0.35, 0.35, -0.55])[None, :, None, None]
    distractor_spectrum = torch.tensor([-0.45, 0.35, 0.35, 1.0])[None, :, None, None]
    distractor = (size - 1) - centres
    frames = []
    for frame in range(length):
        current = target[:, frame]
        distance = (rows - current[:, 0, None, None, None]).square() + (
            cols - current[:, 1, None, None, None]
        ).square()
        signal = (-distance / (2 * 0.9**2)).exp()
        other = distractor + translation[:, frame]
        # Keep the distractor away from the new target in the change fixture.
        if condition == "local_change_stale_history":
            other = torch.stack((centres[:, 0], (size - 1) - centres[:, 1]), 1)
        other_distance = (rows - other[:, 0, None, None, None]).square() + (
            cols - other[:, 1, None, None, None]
        ).square()
        field = 0.3 + contrast * (
            signal * target_spectrum * visible[:, frame, None, None, None]
            + (-other_distance / (2 * 1.1**2)).exp() * distractor_spectrum
        )
        frame_generator = torch.Generator().manual_seed(seed * 10000 + frame)
        noise = torch.randn(batch_size, 4, size, size, generator=frame_generator) * noise_sigma
        # Calibrated Gaussian readout is not clipped; its declared variance is
        # therefore exact, including possible small negative read-noise samples.
        frames.append(field + noise)
    values = torch.stack(frames, 1)
    return ContentBatch(
        values=values,
        variance=torch.full_like(values, max(noise_sigma**2, 1e-10)),
        valid=torch.ones_like(values, dtype=torch.bool),
        target_xy=target,
        visible=visible,
        translations=translation,
        condition=condition,
        event_frame=event_frame,
        seed=seed,
    )

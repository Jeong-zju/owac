"""One-dimensional same-semantics example of the new analytic contract.

The old :mod:`owac.frameworks.n01.sequence1d` adapter exists to show that the N01
support rules do not depend on two-dimensional convolution: a one-dimensional
heteroscedastic sequence, the same unique-frame buffer and the same
merge/split/retract semantics.  That entry point and its tests are kept as the
historical reference; this module re-expresses the same shape in the P1
vocabulary, so the new contract has a 1-D adaptation example of its own:

* one exposure is a row of ``L`` samples with per-sample declared variance;
* identities are ``SourceKey`` values on a degenerate ``y = 0`` grid, so the
  second spatial axis contributes no structure (exactly the old module's
  ``1 x L`` plane argument);
* samples are observed into a :class:`~owac.frameworks.n01_full.ledger.MeasurementLedger`,
  a domain is built with :mod:`~owac.frameworks.n01_full.domains`, and the fit is
  the same ``fit_structure`` used for 2-D data -- with the affine basis reduced to
  ``phi = [1, x]`` by the declared ``dimension=1``;
* repeated frames do not add measurements, missing/bad/saturated samples never
  become zero observations, and retraction removes a sample from the domain
  without touching the raw record.

Only the input stem, the evidence blocks and the readout of the old adapter were
one-dimensional; here the same is true of the *geometry*: nothing in the analytic
path assumes an image grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full.domains import VersionedDomain, make_domain
from owac.frameworks.n01_full.ledger import MeasurementLedger, ObserveResult
from owac.frameworks.n01_full.statistics import NoiseModel, fit_structure

__all__ = [
    "Sequence1DSpec",
    "make_sequence",
    "sequence_frames",
    "sequence_sources",
    "sequence_noise_model",
    "observe_sequence",
    "sequence_domain",
    "sequence1d_example",
]


@dataclass(frozen=True)
class Sequence1DSpec:
    """One 1-D heteroscedastic sequence: values, per-sample variance and validity."""

    values: torch.Tensor  # [T, L] calibrated measurement
    variance: torch.Tensor  # [T, L] declared per-sample variance
    valid: torch.Tensor  # [T, L]
    bad: torch.Tensor  # [T, L]
    saturated: torch.Tensor  # [T, L]
    frame_ids: tuple[int, ...]
    episode_id: str = "sequence-episode"
    camera_id: str = "sequence"
    time_unit: str = "frame"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("values", "variance", "valid", "bad", "saturated"):
            tensor = getattr(self, name)
            if tensor.ndim != 2:
                raise ValueError(f"Sequence1DSpec.{name} must be [T, L]")
        frames, length = self.values.shape
        if len(self.frame_ids) != frames:
            raise ValueError("Sequence1DSpec.frame_ids must have one id per exposure")
        if tuple(self.variance.shape) != (frames, length):
            raise ValueError("Sequence1DSpec.variance must match the values")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError(
                "two exposures cannot share one frame identity; a repeated transmission is "
                "the same measurement, not a new one"
            )

    @property
    def frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        return int(self.values.shape[1])

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "synthetic_1d_heteroscedastic_sequence",
            "frames": self.frames,
            "length": self.length,
            "frame_ids": list(self.frame_ids),
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "seed": self.seed,
            "noise": "declared per-sample variance (heteroscedastic along the sequence)",
        }


def make_sequence(
    *,
    frames: int = 4,
    length: int = 8,
    seed: int = 0,
    slope: float = 0.5,
    intercept: float = 1.0,
    noise_scale: float = 0.01,
    heteroscedastic_ratio: float = 4.0,
    dead_samples: int = 0,
    saturated_samples: int = 0,
    episode_id: str = "sequence-episode",
    camera_id: str = "sequence",
) -> Sequence1DSpec:
    """Deterministic 1-D sequence with a declared, non-uniform variance profile."""
    if frames < 1 or length < 2:
        raise ValueError("the 1-D example needs at least one frame of two samples")
    generator = torch.Generator().manual_seed(seed)
    coordinates = torch.arange(length, dtype=torch.float64)
    field = intercept + slope * coordinates
    values = torch.empty((frames, length), dtype=torch.float64)
    variance = torch.empty((frames, length), dtype=torch.float64)
    boost = torch.ones(length, dtype=torch.float64)
    boost[length // 2 :] = heteroscedastic_ratio
    for frame in range(frames):
        sample = field + noise_scale * torch.randn(length, generator=generator) * boost.sqrt()
        values[frame] = sample
        variance[frame] = (noise_scale * boost) ** 2
    valid = torch.ones((frames, length), dtype=torch.bool)
    bad = torch.zeros((frames, length), dtype=torch.bool)
    saturated = torch.zeros((frames, length), dtype=torch.bool)
    flat = length * (frames - 1)
    for offset in range(dead_samples):
        bad.view(-1)[(offset * 3) % (frames * length)] = True
    for offset in range(saturated_samples):
        saturated.view(-1)[(flat + offset) % (frames * length)] = True
    values = values.masked_fill(bad | saturated, 0.0)
    return Sequence1DSpec(
        values=values,
        variance=variance,
        valid=valid,
        bad=bad,
        saturated=saturated,
        frame_ids=tuple(range(frames)),
        episode_id=episode_id,
        camera_id=camera_id,
        seed=seed,
    )


def sequence_frames(spec: Sequence1DSpec) -> tuple[MeasurementFrame, ...]:
    """One camera frame per exposure: the 1-D row lives on a ``y = 0`` grid."""
    frames = []
    for index, frame_id in enumerate(spec.frame_ids):
        frames.append(
            MeasurementFrame(
                values=spec.values[index].reshape(1, 1, 1, spec.length),
                valid_mask=spec.valid[index].reshape(1, 1, 1, spec.length),
                bad_mask=spec.bad[index].reshape(1, 1, 1, spec.length),
                saturated_mask=spec.saturated[index].reshape(1, 1, 1, spec.length),
                episode_ids=[spec.episode_id],
                frame_ids=[frame_id],
                camera_id=spec.camera_id,
                timestamp=torch.tensor([float(frame_id)]),
                time_unit=spec.time_unit,
                new_measurement=torch.tensor([True]),
                metadata={"sample_axis": "1d", "variance_source": "declared_per_sample"},
            )
        )
    return tuple(frames)


def sequence_sources(spec: Sequence1DSpec) -> tuple[Any, ...]:
    """The ``SourceKey`` of every sample in the sequence, in frame-major order."""
    from owac.frameworks.n01_full.contracts import SourceKey

    return tuple(
        SourceKey(
            "sequence-1d",
            spec.episode_id,
            spec.camera_id,
            frame_id,
            0,
            0,
            position,
        )
        for frame_id in spec.frame_ids
        for position in range(spec.length)
    )


def sequence_noise_model(
    spec: Sequence1DSpec, *, version: str = "sequence1d-declared-v1"
) -> NoiseModel:
    keys = sequence_sources(spec)
    variances = [float(item) for row in spec.variance for item in row]
    return NoiseModel.diagonal(dict(zip(keys, variances, strict=True)), version=version)


def observe_sequence(
    spec: Sequence1DSpec, ledger: MeasurementLedger, *, repeat_first: bool = False
) -> tuple[ObserveResult, ...]:
    """Observe the sequence; optionally re-send the first frame to show idempotence."""
    outcomes = [ledger.observe(frame) for frame in sequence_frames(spec)]
    if repeat_first:
        outcomes.append(ledger.observe(sequence_frames(spec)[0]))
    return tuple(outcomes)


def sequence_domain(
    ledger: MeasurementLedger,
    spec: Sequence1DSpec,
    *,
    domain_id: str = "sequence",
    basis: str = "phase_local_affine",
    decision_version: str = "sequence1d-dv1",
    mapping_version: str = "sequence1d-mv1",
    noise_version: str | None = "sequence1d-declared-v1",
) -> VersionedDomain:
    """A 1-D domain over every valid sample still present in the ledger."""
    evidence = [key for key in sequence_sources(spec) if ledger.has_source(key)]
    return make_domain(
        domain_id,
        reference_chart="sequence-1d",
        basis=basis,
        phase=0,
        evidence_source_ids=evidence,
        temporal_segment_id="segment-0",
        decision_version=decision_version,
        mapping_version=mapping_version,
        noise_version=noise_version,
    )


def sequence1d_example(
    *, frames: int = 4, length: int = 8, seed: int = 0, basis: str = "phase_local_affine"
) -> dict[str, Any]:
    """Run the 1-D adaptation example and return a receipt-friendly summary.

    The example is deliberately small and deterministic: it observes a repeated
    frame, fits the same per-phase GLS on a 1-D affine basis, retracts one sample,
    and records the numbers that a reader can re-derive by hand.
    """
    spec = make_sequence(frames=frames, length=length, seed=seed)
    ledger = MeasurementLedger("sequence-1d")
    outcomes = observe_sequence(spec, ledger, repeat_first=True)
    duplicate = outcomes[-1]
    keys = sequence_sources(spec)
    noise = sequence_noise_model(spec)
    domain = sequence_domain(ledger, spec, basis=basis)
    fit = fit_structure(domain, ledger, None, noise, dimension=1)
    probe = keys[0]
    coord = ledger.sample_coordinates(probe)
    mean, variance = fit.predict(0, coord)
    retracted = sequence_domain(ledger.without([probe]), spec, basis=basis, domain_id="sequence")
    fit_after = fit_structure(retracted, ledger.without([probe]), None, noise, dimension=1)
    mean_after, variance_after = fit_after.predict(0, coord)
    return {
        "spec": spec.spec(),
        "basis": basis,
        "observed_frames": len(outcomes),
        "unique_frames": ledger.unique_frame_count(camera_id=spec.camera_id),
        "duplicate_measurements_added": duplicate.new_samples,
        "independent_observation_count": fit.estimate.independent_observation_count,
        "condition_number": fit.condition_number,
        "theta": [float(item) for item in fit.estimate.theta],
        "probe_coordinate": [coord[0], coord[1]],
        "probe_mean": mean,
        "probe_variance": variance,
        "retracted_observation_count": fit_after.estimate.independent_observation_count,
        "retracted_probe_mean": mean_after,
        "retracted_probe_variance": variance_after,
        "variance_delta": variance_after - variance,
        "raw_record_untouched": (ledger.get_source(probe).raw_value == spec.values[0, 0].item()),
        "support_version": domain.support_version,
        "retracted_support_version": retracted.support_version,
    }

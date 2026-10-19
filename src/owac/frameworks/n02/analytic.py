"""Explicit two-hypothesis measurement update and the exact Bayes reference.

This module is the first, hand-checkable milestone of N02. It fixes a tiny
measurement problem where every quantity can be written down:

* a scene has one binary latent geometry ``g in {0, 1}`` (for example an opening
  that faces left or right),
* the declared forward operator ``M(g)`` maps that geometry to the *raw* value the
  sensor would read,
* a measurement is ``y = M(g) + noise`` with a declared variance, and
* the two explanations start from a declared prior odds, not a learned one.

The exact posterior is then

``posterior_k ∝ prior_k * exp(-sum_i 0.5 * (y_i - M(g_k))^2 / sigma_i^2)``

which is the same expression N02 uses, with the Huber cap switched off and the
``log variance`` term constant across candidates. Anything the recurrent field
does must reproduce this number in the static case, must not grow with the number
of internal cycles on a frozen measurement set, and must change only when a *new*
measurement arrives.

The two forward operators are intentionally simple and named:

``opening_direction``
    A left-facing opening turns the sample coordinate into ``1 - x`` before the
    response is read, so the two hypotheses predict mirrored raw profiles.
``handle_side``
    A handle on one side adds a localised bump at the corresponding coordinate.

Neither operator renders an image: it returns raw values at the sampled
coordinates, which is exactly what the measurement likelihood consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from owac.core.encoder_contract_n02 import MeasurementFrame
from owac.frameworks.n02.config import N02Config, RevisionConfig
from owac.frameworks.n02.hypotheses import neighbour_validity
from owac.frameworks.n02.measurement import gaussian_nll
from owac.frameworks.n02.state import (
    HypothesisState,
    MeasurementBuffer,
    MeasurementEntry,
)


@dataclass(slots=True)
class AnalyticGeometry:
    """Declared latent geometry of the explicit two-hypothesis problem."""

    kind: str = "opening_direction"  # opening_direction | handle_side
    extent: float = 1.0
    bump_width: float = 0.25
    bump_height: float = 0.8
    curvature: float = 0.0

    def names(self) -> tuple[str, ...]:
        if self.kind == "opening_direction":
            return ("left", "right")
        if self.kind == "handle_side":
            return ("handle_left", "handle_right")
        raise ValueError(f"unknown analytic geometry kind: {self.kind!r}")

    @property
    def K(self) -> int:
        return len(self.names())


def forward_operator(
    hypothesis: int,
    coordinates: torch.Tensor,
    geometry: AnalyticGeometry,
    *,
    base: float = 0.5,
) -> torch.Tensor:
    """The declared measurement-domain forward operator ``M(g_k)``.

    ``coordinates`` is ``[N]`` in the normalised sampling range ``[0, extent]``.
    The result is the noiseless raw reading at those coordinates.
    """
    if hypothesis not in (0, 1):
        raise ValueError("the explicit example has exactly two hypotheses")
    x = coordinates / geometry.extent
    if geometry.kind == "opening_direction":
        # A right-facing opening reads the mirrored profile: x -> 1 - x.
        mirror = (1.0 - x) if hypothesis == 1 else x
        profile = base + 0.4 * mirror
    elif geometry.kind == "handle_side":
        centre = 0.25 if hypothesis == 0 else 0.75
        distance = (x - centre) / geometry.bump_width
        profile = base + geometry.bump_height * torch.exp(-0.5 * distance**2)
    else:
        raise ValueError(f"unknown analytic geometry kind: {geometry.kind!r}")
    if geometry.curvature:
        profile = profile + geometry.curvature * (x - 0.5) ** 2
    return profile


@dataclass(slots=True)
class AnalyticObservation:
    """One real measurement of the explicit problem: value, coordinate, variance."""

    value: float
    coordinate: float
    variance: float
    identity: tuple[str, int, int | None]
    saturated: bool = False
    valid: bool = True

    @property
    def counts(self) -> bool:
        """Saturated samples are excluded by default rather than treated as exact."""
        return bool(self.valid and not self.saturated)


@dataclass(slots=True)
class AnalyticCase:
    """A full explicit problem instance with a known generating hypothesis."""

    config: N02Config
    geometry: AnalyticGeometry
    prior_odds: tuple[float, float]
    observations: list[AnalyticObservation] = field(default_factory=list)
    true_hypothesis: int = 0
    evidence_step: int | None = None
    name: str = ""
    prior_probability: tuple[float, float] = (0.5, 0.5)

    def __post_init__(self) -> None:
        if self.geometry.K != self.config.K and self.config.K != 2:
            raise ValueError(
                "the explicit example needs K=2 (or a config whose K is the geometry's)"
            )
        total = float(sum(self.prior_odds))
        if total <= 0.0:
            raise ValueError("prior odds must be positive")
        self.prior_probability = (self.prior_odds[0] / total, self.prior_odds[1] / total)

    # ------------------------------------------------------------ exact Bayes
    def per_hypothesis_nll(self) -> tuple[float, float]:
        """Sum of ``0.5 * residual^2 / variance`` for each hypothesis."""
        totals = [0.0, 0.0]
        for observation in self.observations:
            if not observation.counts:
                continue
            coordinate = torch.tensor([observation.coordinate], dtype=torch.float64)
            for hypothesis in (0, 1):
                predicted = forward_operator(hypothesis, coordinate, self.geometry)
                residual = float(
                    (torch.tensor(observation.value, dtype=torch.float64) - predicted).item()
                )
                totals[hypothesis] += 0.5 * residual**2 / observation.variance
        return totals[0], totals[1]

    def exact_log_posterior(self) -> tuple[float, float]:
        nll0, nll1 = self.per_hypothesis_nll()
        log_prior = torch.log(torch.tensor(self.prior_probability, dtype=torch.float64))
        # Both NLL terms must stay double precision: building them with the default
        # dtype silently rounded the "exact" reference to float32 and made it
        # disagree with a float64 implementation by ~5e-8.
        logits = torch.stack(
            [
                log_prior[0] - torch.tensor(nll0, dtype=torch.float64),
                log_prior[1] - torch.tensor(nll1, dtype=torch.float64),
            ]
        )
        return tuple(float(value) for value in torch.log_softmax(logits, dim=0))

    def exact_posterior(self) -> tuple[float, float]:
        """Exact posterior in double precision.

        The criterion is evaluated from the declared double-precision
        observations, so this reference is exact up to float64 rounding. A
        comparison against the implementation must use an absolute tolerance no
        tighter than float64 epsilon on the logits (roughly 1e-9 here); the
        float32 measurement buffer is a separate, coarser representation and is
        never the reference for this test.
        """
        log_posterior = self.exact_log_posterior()
        return tuple(
            float(torch.exp(torch.tensor(value, dtype=torch.float64))) for value in log_posterior
        )

    def exact_evidence_count(self) -> int:
        return sum(1 for observation in self.observations if observation.counts)

    def contradiction_flags(self, threshold: float = 2.5) -> tuple[bool, bool]:
        """Which candidate the *latest* measurement refutes, by normalised residual."""
        if not self.observations:
            return (False, False)
        latest = self.observations[-1]
        coordinate = torch.tensor([latest.coordinate], dtype=torch.float64)
        flags = []
        for hypothesis in (0, 1):
            predicted = float(forward_operator(hypothesis, coordinate, self.geometry).item())
            residual = abs(latest.value - predicted) / (latest.variance**0.5)
            flags.append(bool(latest.counts and residual > threshold))
        return flags[0], flags[1]

    def to_frame(self, *, camera_id: str = "cam0", frame_id: int = 0) -> MeasurementFrame:
        """Package the explicit observations as one contract measurement frame.

        Values live on a 1-D grid so the frame is a genuine measurement container
        rather than an image: the explicit example never renders anything.
        """
        values = torch.tensor(
            [[[observation.value] for observation in self.observations]], dtype=torch.float32
        ).reshape(1, 1, 1, -1)
        valid = torch.tensor(
            [[[[observation.counts for observation in self.observations]]]], dtype=torch.bool
        ).reshape(1, 1, 1, -1)
        variance = torch.tensor(
            [[[observation.variance for observation in self.observations]]], dtype=torch.float32
        ).reshape(1, 1, 1, -1)
        coordinates = torch.tensor(
            [observation.coordinate for observation in self.observations], dtype=torch.float32
        ).reshape(1, 1, 1, -1)
        return MeasurementFrame(
            values=values,
            valid_mask=valid,
            sample_xy=torch.stack([torch.zeros_like(coordinates), coordinates], dim=-1),
            frame_ids=(frame_id,),
            episode_ids=(0,),
            camera_id=camera_id,
            metadata={
                "kind": "analytic_two_hypothesis",
                "geometry": self.geometry.kind,
                "variance": variance,
                "units": "analytic_raw",
                "declared_forward_operator": True,
            },
        )


def analytic_entries(case: AnalyticCase) -> list[MeasurementEntry]:
    """Measurement-buffer entries for the case's observations (one frame each)."""
    entries = []
    for index, observation in enumerate(case.observations):
        values = torch.tensor([[[[observation.value]]]], dtype=torch.float32)
        valid = torch.tensor([[[[observation.counts]]]], dtype=torch.bool)
        variance = torch.tensor([[[[observation.variance]]]], dtype=torch.float32)
        coordinate = torch.tensor([[[[observation.coordinate]]]], dtype=torch.float32)
        entries.append(
            MeasurementEntry(
                values=values,
                valid_mask=valid,
                bad_mask=None,
                saturated_mask=torch.tensor([[[[observation.saturated]]]], dtype=torch.bool),
                variance=variance,
                sample_xy=torch.stack([torch.zeros_like(coordinate), coordinate], dim=-1),
                camera_id="cam0",
                frame_id=index,
                episode_id=0,
                timestamp=None,
                time_unit="frame",
                metadata={"units": "analytic_raw"},
            )
        )
    return entries


def build_buffer(case: AnalyticCase, *, max_history: int | None = None) -> MeasurementBuffer:
    """Static (whole-buffer) measurement set for the explicit example."""
    entries = analytic_entries(case)
    buffer = MeasurementBuffer(
        max_history=max_history or max(len(entries), 1),
        mask_history=max_history is not None,
    )
    for entry in entries:
        buffer.add(entry)
    return buffer


def analytic_energy(
    case: AnalyticCase,
    buffer: MeasurementBuffer,
    config: N02Config,
    *,
    robust: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy of the two explicit hypotheses on a frozen measurement set.

    Returns ``(energy, prior_probability)`` as ``[1, 2]`` float64 tensors. The
    energy is evaluated through the *same* :func:`gaussian_nll` the learned path
    uses, on the declared double-precision observations, with the Huber cap off by
    default so it equals the closed-form reference exactly. The buffer decides
    which frames count: a frame whose validity mask is false contributes nothing,
    and every frame contributes at most once.
    """
    robust_mode = config.measurement.robust if robust is None else robust
    valid_frames = [
        index
        for index in range(min(len(buffer.frames), len(case.observations)))
        if buffer.frames[index].valid_mask is None
        or bool(buffer.frames[index].valid_mask.flatten()[0].item())
    ]
    energy = torch.zeros(1, case.geometry.K, dtype=torch.float64)
    for hypothesis in range(case.geometry.K):
        total = torch.zeros((), dtype=torch.float64)
        for index in valid_frames:
            observation = case.observations[index]
            predicted = forward_operator(
                hypothesis,
                torch.tensor([observation.coordinate], dtype=torch.float64),
                case.geometry,
            ).to(torch.float64)
            residual = torch.tensor([[observation.value]], dtype=torch.float64) - predicted.reshape(
                1, 1
            )
            log_variance = torch.log(torch.tensor([[observation.variance]], dtype=torch.float64))
            total = (
                total
                + gaussian_nll(
                    residual,
                    log_variance,
                    robust=robust_mode,
                    huber_delta=config.measurement.huber_delta,
                ).mean()
            )
        energy[0, hypothesis] = total
    prior = torch.tensor([case.prior_probability], dtype=torch.float64)
    return energy, prior


def analytic_posterior(
    case: AnalyticCase,
    buffer: MeasurementBuffer,
    config: N02Config,
    *,
    beta_prior: float | None = None,
    temperature: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The N02 scoring rule applied to the explicit problem.

    ``weights = softmax(-(measurement_energy + beta_prior * prior_energy) / T)``
    with ``prior_energy = -log prior`` and ``T = 1`` reproduces the exact Bayes
    posterior. The returned diagnostics carry the raw pieces so a caller can audit
    the arithmetic instead of trusting one number.
    """
    energy, prior_probability = analytic_energy(case, buffer, config)
    beta = config.beta_prior if beta_prior is None else beta_prior
    temperature = config.temperature if temperature is None else temperature
    if beta == 0.0:
        prior_energy = torch.zeros_like(energy)
    else:
        prior_energy = -torch.log(prior_probability)
    logits = -(energy + beta * prior_energy) / temperature
    weights = torch.softmax(logits, dim=-1)
    diagnostics = {
        "measurement_energy": [float(value) for value in energy[0].tolist()],
        "prior_probability": [float(value) for value in prior_probability[0].tolist()],
        "prior_energy": [float(value) for value in prior_energy[0].tolist()],
        "logits": [float(value) for value in logits[0].tolist()],
        "weights": [float(value) for value in weights[0].tolist()],
        "observations_used": len(buffer.frames),
        "duplicates_skipped": buffer.duplicate_count,
        "temperature": float(temperature),
        "beta_prior": float(beta),
    }
    return weights, diagnostics


def repeated_observation_check(case: AnalyticCase, config: N02Config) -> dict[str, Any]:
    """Submit the same frame twice and prove the likelihood is not doubled.

    The buffer keeps one entry, the duplicate counter advances, and the posterior
    is bit-identical to the single-submission case.
    """
    single = build_buffer(case)
    weights_single, diag_single = analytic_posterior(case, single, config)
    duplicated = build_buffer(case)
    if duplicated.frames:
        duplicated.add(analytic_entries(case)[-1])
    weights_duplicated, diag_duplicated = analytic_posterior(case, duplicated, config)
    identical = bool(torch.equal(weights_single, weights_duplicated))
    return {
        "unique_evidence_single": single.unique_evidence_count,
        "unique_evidence_duplicated": duplicated.unique_evidence_count,
        "duplicates_recorded": duplicated.duplicate_count,
        "posterior_identical": identical,
        "posterior_single": diag_single["weights"],
        "posterior_duplicated": diag_duplicated["weights"],
        "measurement_energy_single": diag_single["measurement_energy"],
        "measurement_energy_duplicated": diag_duplicated["measurement_energy"],
    }


def make_state(config: N02Config, weights: torch.Tensor) -> HypothesisState:
    """A field-free :class:`HypothesisState` carrying only the analytic posterior.

    The explicit case deliberately keeps ``h`` at a 1x1 grid: the point of this
    module is the *update rule*, and adding a spatial field would make the
    hand-checkable arithmetic harder to audit, not more informative.
    """
    batch, K = weights.shape
    device = weights.device
    return HypothesisState(
        h=torch.zeros(batch, 1, K, config.D, 1, 1, device=device),
        measurement_energy=torch.zeros(batch, 1, K, device=device),
        weights=weights.detach().unsqueeze(1),
        prior_energy=torch.zeros(batch, 1, K, device=device),
        relation_energy=torch.zeros(batch, 1, K, device=device),
        total_energy=torch.zeros(batch, 1, K, device=device),
        logits=torch.zeros(batch, 1, K, device=device),
        geometry=torch.zeros(batch, 1, K, config.measurement.geometry_dim, device=device),
        age=torch.zeros(batch, 1, K, device=device),
        times_reset=torch.zeros(batch, 1, K, device=device),
        dependency_valid=neighbour_validity(1, 1, device).view(1, 1, 1, 1, 4),
    )


def delayed_evidence_case(
    config: N02Config,
    *,
    geometry: AnalyticGeometry | None = None,
    prior_odds: tuple[float, float] = (1.0, 1.0),
    true_hypothesis: int = 0,
    delay: int = 3,
    ambiguity_sigma: float = 1.2,
    cue_sigma: float = 0.08,
    seed: int = 0,
    misidentify_coordinate: bool = False,
) -> AnalyticCase:
    """Ambiguous observations, then a distinguishing cue after ``delay`` frames.

    The first ``delay`` measurements are near the point where both hypotheses
    predict the same value (so they carry little information), and the cue
    measurement is taken where the two profiles differ by ``bump_height`` (or the
    mirror separation) with a much smaller variance. ``misidentify_coordinate``
    moves the cue to the *shared* point instead, which produces a genuinely
    unidentifiable sequence for the "stays ambiguous" control.
    """
    geometry = geometry or AnalyticGeometry()
    generator = torch.Generator().manual_seed(seed)
    observations: list[AnalyticObservation] = []
    ambiguous_coordinate = 0.5 * geometry.extent
    for index in range(delay):
        truth = float(
            forward_operator(true_hypothesis, torch.tensor([ambiguous_coordinate]), geometry).item()
        )
        # ``ambiguity_sigma``/``cue_sigma`` below are measurement *variances*, so
        # the sampled noise uses their square root: sigma = sqrt(variance).
        noise = float(torch.randn(1, generator=generator).item()) * (ambiguity_sigma**0.5)
        observations.append(
            AnalyticObservation(
                value=truth + noise,
                coordinate=ambiguous_coordinate,
                variance=ambiguity_sigma,
                identity=("cam0", index, 0),
            )
        )
    if geometry.kind == "opening_direction":
        # Both hypotheses differ at the quarter point; which one the cue supports
        # depends on the true geometry, not on a fixed frame index.
        cue_coordinate = 0.25 * geometry.extent
    else:
        cue_coordinate = (0.25 if true_hypothesis == 0 else 0.75) * geometry.extent
    if misidentify_coordinate:
        cue_coordinate = ambiguous_coordinate
    cue_truth = float(
        forward_operator(true_hypothesis, torch.tensor([cue_coordinate]), geometry).item()
    )
    cue_noise = float(torch.randn(1, generator=generator).item()) * (cue_sigma**0.5)
    observations.append(
        AnalyticObservation(
            value=cue_truth + cue_noise,
            coordinate=cue_coordinate,
            variance=cue_sigma,
            identity=("cam0", delay, 0),
        )
    )
    return AnalyticCase(
        config=config,
        geometry=geometry,
        prior_odds=prior_odds,
        observations=observations,
        true_hypothesis=true_hypothesis,
        evidence_step=delay,
        name=f"delayed_evidence_{geometry.kind}_seed{seed}",
    )


def make_config(
    *,
    K: int = 2,
    cycles: int = 4,
    history: int = 4,
    beta_prior: float = 1.0,
    temperature: float = 1.0,
    robust: str = "gaussian",
    revision: RevisionConfig | None = None,
) -> N02Config:
    """A small explicit-problem configuration; no learned component is used."""
    config = N02Config(
        name="analytic",
        K=K,
        D=8,
        cycles=cycles,
        history=history,
        beta_prior=beta_prior,
        temperature=temperature,
        stem_width=8,
        cell_kernel=3,
        revision=revision or RevisionConfig(),
    )
    config.measurement.robust = robust
    config.measurement.predict_variance = False
    return config

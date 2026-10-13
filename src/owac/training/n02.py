"""Training entry point for the N02 measurement-constrained recurrent field.

Command (the target command from the implementation prompt)::

    uv run --locked --extra encoder python -m owac.training.n02 \\
        --config configs/encoders/n02/tiny.json --synthetic --steps 200 \\
        --output outputs/n02-smoke

Objective
---------
One training step consumes one batch of ``B`` event sequences of ``T`` frames and
calls the *same* :meth:`N02Encoder.step` used at inference, frame by frame, with
the state detached at every boundary (the framework keeps a fresh graph per step,
so no truncated-BPTT horizon has to be configured). The loss is

``L = w_m*L_measurement + w_g*L_geometry_set + w_r*L_revision
       + w_d*L_diversity + w_mix*L_mixture``

with ``w_*`` read from ``config.loss_weights``. Every term is reported separately
(unweighted and weighted) together with its effective sample count.

Normalisation, stated once and used for every term
--------------------------------------------------
* ``E_k`` is the encoder's own ``representation.measurement_energy`` for
  candidate ``k``: the robust NLL of the *frozen* current measurement, reduced
  over its valid samples. Every energy-based term is divided by ``N_obs`` --
  the number of likelihood observations scored in this step, reported as
  ``likelihood_observations``. The causal dynamic path uses one current
  likelihood and carries earlier evidence once through the frozen prior.
* Each term is then a mean over the batch elements that are *valid* for it, and
  the batch-level term is the mean over the steps in which it was active. A term
  with no valid sample in a step is skipped for that step and the reason is
  recorded; it is never silently replaced by zero.

The five terms
--------------
``L_measurement``
    Mean measurement-space NLL over every compatible slot: all candidates before
    informative RAW, otherwise all slots of the true semantic parity. The anchor
    is the declared two-way task structure (synthetic: ``supervision['value_over_time']``;
    analytic: the generating hypothesis index). The hidden geometry enters the
    loss only -- it is never placed in the encoder input.
``L_geometry_set``
    Supervised binary geometry proxy: slots of even/odd parity predict -1/+1
    through the geometry head. A single-hypothesis model predicts the current
    truth proxy. This is a declared semantic surrogate, not full physical geometry.
``L_mixture``
    Cross entropy of the actual recurrent posterior class mass, including the
    frozen prior and once-per-frame withdrawal. Before informative RAW the target
    retains the declared prior (uniform only when uninformative/missing); after
    informative RAW it is the true semantic class. Stable log scores preserve
    gradients for a confidently wrong posterior. Kept separate from geometry.
``L_revision``
    Active only on the post-contradiction steps of independently declared
    sequences (a cue that refutes a misleading prior, or a declared scene
    change). It has two named parts, both means over the valid samples:
    ``wrong_weight = mean w_wrong`` (the refuted explanation must lose weight) and
    ``wrong_energy_gap = mean relu(E_right - E_wrong) / N_obs`` (the refuted
    explanation must fit the measurements worse than the true one).
    ``L_revision = wrong_weight + wrong_energy_gap``.
``L_diversity``
    ``relu(margin - mean_pairwise_normalised_distance(h))``: a hinge that only
    penalises *collapse* of the K candidates (see
    :func:`owac.frameworks.n02.hypotheses.diversity_penalty`), so it does not push
    a single-explanation answer apart. It is active only when the multi-solution
    condition holds: the current measurements remain ambiguous; visible distinguishing cues
    disable this penalty even when K exceeds two.

Sources
-------
``--synthetic`` uses :mod:`owac.frameworks.n02.synthetic` when it is importable
(ambiguous prefix -> cue -> noise/occlusion sequences, prior conditions
correct/misleading/uninformative/missing/stale). Otherwise, and with
``--data-source analytic``, it falls back to
:func:`owac.frameworks.n02.analytic.delayed_evidence_case`, whose two explicit
hypotheses are placed on a one-dimensional measurement grid at their real sample
coordinates. The path actually used is recorded in ``summary.json``.

CPU cost (measured, not projected)
---------------------------------
torch's default intra-op thread count (24-32 here) is a large *loss* on the small
tensors this mechanism uses: a single encoder step was 4.8 s with 8 threads and
85 ms with 1 thread on the same input. ``--threads`` therefore defaults to 1.

The cost driver is the encoder itself, at ``batch * frames`` calls per training
step. Measured on this CPU host with the tiny config (4 cycles, 4 history):

===============================================  ==========  ==================
shape (batch, sequence, frame side)              encoder p50  CPU seconds/step
===============================================  ==========  ==================
(4, 8, 80) -- generator default frame            217 ms      3.06
(2, 8, 64) -- CLI default                        30.5 ms     0.58
(2, 6, 64)                                       28.3 ms     0.40
===============================================  ==========  ==================

So the defaults below are the timed, prompt-compliant shape (sequence length 8):
``--batch-size 2 --sequence-length 8 --frame-size 64``, which runs the documented
``--steps 200`` smoke in about two minutes of CPU. ``--frame-size 80`` (the
generator's own default geometry) at batch 4 costs about 7x more per step; both
numbers are recorded in ``summary.json`` together with the CPU time, the wall
time and the machine load average, so a slow wall time under a shared machine is
attributable rather than mysterious. ``cpu_seconds_per_step`` is the
contention-independent number; ``fraction_of_wall_that_is_cpu`` states how much of
the wall time this process actually received.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from owac.core.encoder_contract import (
    CameraSpec,
    ComputeBudget,
    EmbodimentContext,
    MeasurementFrame,
)
from owac.frameworks.n02.config import N02Config, load_config
from owac.frameworks.n02.hypotheses import diversity_penalty
from owac.frameworks.n02.model import build_encoder
from owac.frameworks.n02.readout import (
    TokenReadout,
    hypothesis_tokens,
    map_hypothesis,
    representation_from_output,
)

#: Number of declared values in the two-way mechanism tasks. Candidate slots
#: ``0..K_ANCHOR-1`` are anchored to them; extra slots are unanchored.
K_ANCHOR = 2

#: Hinge margin of ``L_diversity`` on the mean pairwise normalised candidate
#: distance (which lies in ``[0, 2]``). Only collapse below this margin is
#: penalised, so a single-explanation answer is never forced to split.
DIVERSITY_MARGIN = 0.1

PRIOR_CONDITIONS = ("correct", "misleading", "uninformative", "missing", "stale")

#: One-dimensional measurement grid used by the analytic fallback. It must stay
#: below 16 so the encoder's grid-stride rule keeps the field at the frame's own
#: resolution (a coarser field would make a full-resolution validity mask
#: inconsistent with the buffered values).
ANALYTIC_GRID_WIDTH = 8


@dataclass(slots=True)
class StepBatch:
    """One training batch: ``T`` frames plus the supervision needed by the loss.

    ``anchor[t, b]`` is the candidate slot anchored to the latent value in force
    at step ``t`` for batch element ``b`` (``-1`` = this element carries no
    declared anchor at this step). ``wrong[t, b]`` is the anchored slot of the
    competing declared value. ``post_contradiction[t, b]`` marks steps after a
    contradiction arrived, which is the only place ``L_revision`` is defined.
    """

    values: list[torch.Tensor]
    masks: list[torch.Tensor | None]
    anchor: torch.Tensor
    wrong: torch.Tensor
    post_contradiction: torch.Tensor
    camera: CameraSpec
    source: str
    multi_solution: bool = False
    notes: dict[str, Any] = field(default_factory=dict)
    contexts: list[EmbodimentContext] = field(default_factory=list)
    identifiable_now: torch.Tensor | None = None  # evaluator/loss only, [T,B]
    sample_xy: Any = None  # declared sampling descriptor

    @property
    def num_steps(self) -> int:
        return len(self.values)

    @property
    def batch_size(self) -> int:
        return int(self.anchor.shape[1])


def episode_prior_context(episodes: list[Any], K: int) -> EmbodimentContext:
    """Only the declared prior enters the model; condition/truth stay in the loss."""
    from owac.frameworks.n02.synthetic import values_for_task

    mass = torch.ones(len(episodes), K)
    available = torch.tensor([episode.prior.available for episode in episodes])
    for index, episode in enumerate(episodes):
        if not episode.prior.available or K == 1:
            continue
        value_index = values_for_task(episode.task).index(episode.prior.value)
        p = 1.0 / (1.0 + math.exp(-episode.prior.log_odds))
        for value in range(2):
            slots = torch.arange(K) % 2 == value
            if bool(slots.any()):
                mass[index, slots] = (p if value == value_index else 1.0 - p) / int(slots.sum())
    return EmbodimentContext(hypothesis_prior=mass, hypothesis_prior_available=available)


# ----------------------------------------------------------------------------- data


def _synthetic_module() -> tuple[Any | None, str]:
    try:
        from owac.frameworks.n02 import synthetic
    except ImportError as exc:  # pragma: no cover - only when the module is absent
        return None, f"owac.frameworks.n02.synthetic unavailable: {exc}"
    return synthetic, ""


def _analytic_module() -> tuple[Any | None, str]:
    try:
        from owac.frameworks.n02 import analytic
    except ImportError as exc:  # pragma: no cover - the module ships with the framework
        return None, f"owac.frameworks.n02.analytic unavailable: {exc}"
    return analytic, ""


def validate_synthetic_frame_size(frame_size: int, *, seed: int, split: str) -> None:
    """Fail fast when the requested frame cannot hold the generator's geometry.

    The generator's own requirement is stated in ``SyntheticGenerator.centre_bounds``:
    the arc centre must be more than ``cue_clearance`` pixels from every edge, so a
    frame below roughly 41 px cannot carry the two symmetric hypotheses at all. The
    smallest *legal* even size is found by probing the generator once, so the error
    message names the real requirement instead of failing one step into training.
    """
    synthetic, reason = _synthetic_module()
    if synthetic is None:
        raise RuntimeError(reason)
    if frame_size % 2:
        raise SystemExit(
            f"--frame-size {frame_size} is odd; the generator needs an even side so a "
            "2x2 CFA tile is complete"
        )

    def attempt(side: int) -> str:
        options = synthetic.SyntheticOptions(height=side, width=side, channels=3)
        generator = synthetic.SyntheticGenerator(options, seed=seed, split=split)
        try:
            generator.generate(num_frames=4, episode_index=0, seed=seed)
        except ValueError as exc:
            return str(exc)
        return ""

    problem = attempt(frame_size)
    if not problem:
        return
    legal: int | None = None
    for candidate in range(frame_size + 2, 201, 2):
        if not attempt(candidate):
            legal = candidate
            break
    raise SystemExit(
        f"--frame-size {frame_size} is not usable by the synthetic generator: {problem}"
        + (
            f"; the smallest even side that works is {legal}. Use --frame-size {legal} "
            "or larger (the generator's own default is 80)."
            if legal is not None
            else "; no even side up to 200 works, which indicates a generator problem."
        )
    )


def synthetic_batch(
    config: N02Config,
    rng: np.random.Generator,
    *,
    batch_size: int,
    num_frames: int,
    frame_size: int,
    seed: int,
    split: str,
    task: str = "opening_direction",
) -> StepBatch:
    """Draw ``batch_size`` synthetic event sequences and stack them per step."""
    synthetic, reason = _synthetic_module()
    if synthetic is None:
        raise RuntimeError(reason)
    options = synthetic.SyntheticOptions(
        height=frame_size,
        width=frame_size,
        channels=3,
        input_mode=config.measurement.input_mode,
        cfa_pattern=config.measurement.cfa_pattern,
        noise_sigma=0.01,
        occlusion_schedule="linear",
        identifiability=0.8,
    )
    generator = synthetic.SyntheticGenerator(options, seed=seed, split=split)
    values_by_task = synthetic.values_for_task(task)
    anchors = torch.full((num_frames, batch_size), -1, dtype=torch.long)
    wrong = torch.full((num_frames, batch_size), -1, dtype=torch.long)
    post = torch.zeros(num_frames, batch_size, dtype=torch.bool)
    conditions: list[str] = []
    identifiable_flags: list[bool] = []
    contradiction_steps: list[int] = []
    episodes = []
    identifiable_now = torch.zeros(num_frames, batch_size, dtype=torch.bool)
    for index in range(batch_size):
        condition = PRIOR_CONDITIONS[int(rng.integers(0, len(PRIOR_CONDITIONS)))]
        episode_index = int(rng.integers(0, 1_000_000))
        identifiable = bool(rng.random() < 0.8)
        episode = generator.generate(
            task=task,
            prior_condition=condition,
            num_frames=num_frames,
            episode_index=episode_index,
            identifiable=identifiable,
            seed=seed,
        )
        episodes.append(episode)
        conditions.append(condition)
        identifiable_flags.append(bool(episode.identifiable))
        identifiable_now[:, index] = torch.tensor(
            generator.measurement_quality(episode)["identifiable_now"]
        )
        for step in range(num_frames):
            anchor = values_by_task.index(episode.value_over_time[step])
            anchors[step, index] = anchor
            wrong[step, index] = 1 - anchor
        arrival = int(episode.evidence_arrival_step)
        change = int(episode.geometry_change_step)
        declares_contradiction = episode.supervision["prior_is_correct"] is False
        if declares_contradiction:
            start = min(arrival, change) if change >= 0 else arrival
        elif change >= 0:
            start = change
        else:
            start = -1
        contradiction_steps.append(start)
        if start >= 0:
            post[start:, index] = identifiable_now[start:, index]
    values = [
        torch.from_numpy(
            np.stack([episodes[index].frames[step] for index in range(batch_size)])
        ).to(torch.float32)
        for step in range(num_frames)
    ]
    camera = CameraSpec(
        camera_id="synthetic_measurement",
        height=values[0].shape[-2],
        width=values[0].shape[-1],
        channels=values[0].shape[1],
        input_mode=config.measurement.input_mode,
        cfa_pattern=config.measurement.cfa_pattern,
    )
    return StepBatch(
        values=values,
        masks=[None] * num_frames,
        anchor=anchors,
        wrong=wrong,
        post_contradiction=post,
        camera=camera,
        source="synthetic",
        multi_solution=not all(identifiable_flags),
        contexts=[episode_prior_context(episodes, config.K)] * num_frames,
        identifiable_now=identifiable_now,
        sample_xy=torch.from_numpy(episodes[0].sample_xy),
        notes={
            "task": task,
            "prior_conditions": conditions,
            "prior_log_odds": [
                None if not e.prior.available else e.prior.log_odds for e in episodes
            ],
            "prior_sampling": episodes[0].metadata["prior_sampling"],
            "measurement_quality": "physical final-RAW templates; expected log BF >= log(19)",
            "identifiable": identifiable_flags,
            "contradiction_steps": contradiction_steps,
            "frame_size": frame_size,
            "channel_names": ["ch0", "ch1", "ch2"],
            "valid_mask": (
                "omitted: every generated frame is fully valid, and a sensor-resolution "
                "validity mask does not survive the encoder's grid resampling"
            ),
        },
    )


def analytic_batch(
    config: N02Config,
    rng: np.random.Generator,
    *,
    batch_size: int,
    num_frames: int,
    seed: int,
) -> StepBatch:
    """Build the explicit two-hypothesis fallback batch on a 1-D measurement grid.

    ``delayed_evidence_case`` places its observations at real sample coordinates;
    each step's frame carries the observation value **only at its own sampling
    bin**, and the validity mask marks exactly that bin, so the measurement
    likelihood is evaluated where the sensor actually sampled. Post-cue steps are
    appended from the same declared geometry so the revision term sees more than
    one post-contradiction sample; the appended measurements use the cue
    coordinate and the generating hypothesis's own predicted value.
    """
    analytic, reason = _analytic_module()
    if analytic is None:
        raise RuntimeError(reason)
    if ANALYTIC_GRID_WIDTH >= 16:
        raise ValueError("ANALYTIC_GRID_WIDTH must stay below 16 (encoder grid-stride rule)")
    case_config = analytic.make_config(
        K=2,
        cycles=config.cycles,
        history=config.history,
        beta_prior=config.beta_prior,
        temperature=config.temperature,
        robust="gaussian",
    )
    width = ANALYTIC_GRID_WIDTH
    delay = max(2, num_frames // 2)
    values = [torch.zeros(batch_size, 1, 1, width) for _ in range(num_frames)]
    masks = [torch.zeros(batch_size, 1, 1, width, dtype=torch.bool) for _ in range(num_frames)]
    anchors = torch.full((num_frames, batch_size), -1, dtype=torch.long)
    wrong = torch.full((num_frames, batch_size), -1, dtype=torch.long)
    post = torch.zeros(num_frames, batch_size, dtype=torch.bool)
    priors: list[tuple[float, float]] = []
    truths: list[int] = []
    for index in range(batch_size):
        truth = int(rng.integers(0, 2))
        misleading = bool(rng.random() < 0.5)
        odds = [0.9, 0.1] if truth == 0 else [0.1, 0.9]
        if misleading:
            odds = odds[::-1]
        elif rng.random() < 0.3:
            odds = [1.0, 1.0]
        episode_seed = int(rng.integers(0, 1_000_000))
        case = analytic.delayed_evidence_case(
            case_config,
            prior_odds=(odds[0], odds[1]),
            true_hypothesis=truth,
            delay=delay,
            seed=episode_seed,
        )
        cue_coordinate = (
            0.25 * case.geometry.extent
            if case.geometry.kind == "opening_direction"
            else (0.25 if truth == 0 else 0.75) * case.geometry.extent
        )
        cue_value = float(
            analytic.forward_operator(
                truth, torch.tensor([cue_coordinate], dtype=torch.float64), case.geometry
            ).item()
        )
        generator = torch.Generator().manual_seed(episode_seed + 1)
        while len(case.observations) < num_frames:
            noise = float(torch.randn(1, generator=generator).item()) * math.sqrt(0.08)
            case.observations.append(
                analytic.AnalyticObservation(
                    value=cue_value + noise,
                    coordinate=cue_coordinate,
                    variance=0.08,
                    identity=("cam0", len(case.observations), 0),
                )
            )
        contradiction = int(case.evidence_step if misleading else -1)
        for step, observation in enumerate(case.observations[:num_frames]):
            bin_index = int(round(observation.coordinate / case.geometry.extent * (width - 1)))
            bin_index = min(max(bin_index, 0), width - 1)
            values[step][index, 0, 0, bin_index] = float(observation.value)
            masks[step][index, 0, 0, bin_index] = True
            anchors[step, index] = truth
            wrong[step, index] = 1 - truth
            if contradiction >= 0 and step >= contradiction:
                post[step, index] = True
        priors.append((odds[0], odds[1]))
        truths.append(truth)
    camera = CameraSpec(
        camera_id="analytic_measurement",
        height=1,
        width=width,
        channels=1,
        input_mode="bayer_cfa",
        cfa_pattern="RGGB",
    )
    return StepBatch(
        values=values,
        masks=masks,
        anchor=anchors,
        wrong=wrong,
        post_contradiction=post,
        camera=camera,
        source="analytic",
        contexts=[
            EmbodimentContext(
                hypothesis_prior=torch.tensor(
                    [[odds[slot % 2] for slot in range(config.K)] for odds in priors]
                )
            )
        ]
        * num_frames,
        identifiable_now=torch.arange(num_frames)[:, None].expand(-1, batch_size) >= delay,
        multi_solution=config.K > K_ANCHOR,
        notes={
            "geometry": "opening_direction (exact two-hypothesis reference)",
            "grid_width": width,
            "prior_odds": priors,
            "true_hypothesis": truths,
            "input_mode": (
                "bayer_cfa (forced: the analytic reference has one measurement channel; "
                "the config's rawvla_rgb3 head expects three)"
            ),
            "appended_post_cue_observations": num_frames - (delay + 1),
        },
    )


def build_batch(
    config: N02Config,
    rng: np.random.Generator,
    *,
    source: str,
    batch_size: int,
    num_frames: int,
    frame_size: int,
    seed: int,
    split: str,
) -> StepBatch:
    if source == "synthetic":
        return synthetic_batch(
            config,
            rng,
            batch_size=batch_size,
            num_frames=num_frames,
            frame_size=frame_size,
            seed=seed,
            split=split,
        )
    if source == "analytic":
        return analytic_batch(config, rng, batch_size=batch_size, num_frames=num_frames, seed=seed)
    raise ValueError(f"unknown data source: {source!r}")


# ----------------------------------------------------------------------------- loss


@dataclass(slots=True)
class TermAccumulator:
    """Sum/count bookkeeping so every term reports its own effective sample count."""

    total: float = 0.0
    samples: int = 0
    active_steps: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def add(self, value: float, samples: int) -> None:
        self.total += float(value)
        self.samples += int(samples)
        self.active_steps += 1

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    @property
    def mean(self) -> float | None:
        return None if self.active_steps == 0 else self.total / self.active_steps


def step_terms(
    representation: Any,
    *,
    anchor: torch.Tensor,
    wrong: torch.Tensor,
    post_contradiction: torch.Tensor,
    observations: int,
    diversity_active: bool,
    diversity_margin: float = DIVERSITY_MARGIN,
    identifiable_now: torch.Tensor | None = None,
    ambiguity_prior: torch.Tensor | None = None,
) -> dict[str, Any]:
    """All five loss terms for one encoder step, each with its valid-sample mask.

    Returns a dict of tensors (``None`` where the term is not defined for this
    step) plus the integer sample counts and the skip reasons.
    """
    views = representation.weights.shape[1]
    energy = representation.measurement_energy.reshape(
        representation.weights.shape[0], views, -1
    ).mean(dim=1)
    weights = representation.weights.mean(dim=1)
    batch, num_candidates = weights.shape
    per_observation = max(1, int(observations))
    scaled = energy / float(per_observation)
    anchor, wrong, post_contradiction = (
        anchor.to(energy.device),
        wrong.to(energy.device),
        post_contradiction.to(energy.device),
    )
    result: dict[str, Any] = {"counts": {}, "skipped": {}, "observations": per_observation}

    valid_anchor = (anchor >= 0) & (anchor < num_candidates)
    if num_candidates == 1:
        valid_anchor = anchor >= 0
    if identifiable_now is None:
        identifiable_now = post_contradiction
    identifiable_now = identifiable_now.to(energy.device)
    slots = torch.arange(num_candidates, device=energy.device) % 2
    compatible = (~identifiable_now[:, None]) | (slots[None, :] == anchor[:, None])
    if num_candidates == 1:
        compatible = torch.ones_like(compatible)
    if valid_anchor.any():
        selected = (scaled * compatible).sum(-1) / compatible.sum(-1).clamp_min(1)
        result["measurement"] = selected[valid_anchor].mean()
        result["counts"]["measurement"] = int(valid_anchor.sum())
    else:
        result["measurement"] = None
        result["counts"]["measurement"] = 0
        result["skipped"]["measurement"] = "no batch element carries a declared anchor"

    # Binary geometry is a declared mechanism proxy (opening/handle side), not
    # an alias for measurement reconstruction. Slots alternate the two possible
    # geometries, including redundant particles when K>2.
    slot_geometry = (torch.arange(num_candidates, device=energy.device) % 2) * 2.0 - 1.0
    geometry_target = slot_geometry.expand_as(representation.geometry[..., 0])
    if num_candidates == 1:
        geometry_target = (anchor * 2.0 - 1.0)[:, None, None].expand_as(geometry_target)
    result["geometry_set"] = F.mse_loss(representation.geometry[..., 0], geometry_target)
    result["counts"]["geometry_set"] = batch
    result["hard_min_energy"] = scaled.min(dim=-1).values.mean()

    valid_wrong = post_contradiction & (wrong >= 0) & (wrong < num_candidates) & valid_anchor
    if num_candidates == 1:
        valid_wrong = torch.zeros_like(valid_wrong)
    if valid_wrong.any():
        slots = torch.arange(num_candidates, device=energy.device) % 2
        wrong_set = slots[None, :] == wrong[:, None]
        true_set = slots[None, :] == anchor[:, None]
        wrong_weight = (weights * wrong_set).sum(-1)[valid_wrong]

        def class_energy(selected):
            return (
                -torch.logsumexp((-scaled).masked_fill(~selected, -torch.inf), -1)
                + selected.sum(-1).clamp_min(1).log()
            )

        gap = F.relu(class_energy(true_set) - class_energy(wrong_set))[valid_wrong]
        result["revision_wrong_weight"] = wrong_weight.mean()
        result["revision_energy_gap"] = gap.mean()
        result["revision"] = result["revision_wrong_weight"] + result["revision_energy_gap"]
        result["counts"]["revision"] = int(valid_wrong.sum())
    else:
        result["revision"] = None
        result["revision_wrong_weight"] = None
        result["revision_energy_gap"] = None
        result["counts"]["revision"] = 0
        result["skipped"]["revision"] = (
            "no post-contradiction sample with a declared competing candidate"
        )

    # Supervise the actual recurrent posterior, aggregating redundant slots by
    # semantic class. Before informative RAW arrives retain the declared prior;
    # never force an informative prior to uniform, or reveal the future label.
    if ambiguity_prior is None:
        ambiguity_prior = torch.full_like(weights, 1.0 / num_candidates)
    else:
        ambiguity_prior = ambiguity_prior.to(weights).detach()
        if ambiguity_prior.shape != weights.shape:
            raise ValueError("ambiguity_prior must be [B,K]")
        ambiguity_prior = ambiguity_prior / ambiguity_prior.sum(-1, keepdim=True).clamp_min(1e-12)
    class_weights = torch.stack([(weights * (slots == c)).sum(-1) for c in range(2)], -1)
    posterior_logits = getattr(representation, "posterior_logits", None)
    if posterior_logits is None:
        class_logp = class_weights.clamp_min(1e-12).log()
    elif num_candidates == 1:
        class_logp = posterior_logits.mean(dim=1).expand(-1, 2) * 0
    else:
        # These are the exact scores producing representation.weights, including
        # the frozen prior and withdrawal. Log space prevents a confidently wrong
        # posterior from losing its training gradient through softmax underflow.
        slot_logp = F.log_softmax(posterior_logits, -1)
        class_logp = torch.stack(
            [
                torch.logsumexp(slot_logp.masked_fill((slots != c)[None, None], -torch.inf), -1)
                for c in range(2)
            ],
            -1,
        )
        class_logp = torch.logsumexp(class_logp, 1) - math.log(views)
    prior_classes = torch.stack([(ambiguity_prior * (slots == c)).sum(-1) for c in range(2)], -1)
    if num_candidates == 1:
        target_classes = prior_classes
    else:
        target_classes = F.one_hot(anchor.clamp(0, 1), 2).to(weights)
        target_classes = torch.where(identifiable_now[:, None], target_classes, prior_classes)
    result["mixture"] = (
        -(target_classes[valid_anchor] * class_logp[valid_anchor]).sum(-1).mean()
        if valid_anchor.any()
        else None
    )
    result["counts"]["mixture"] = int(valid_anchor.sum())

    if diversity_active:
        distance, collapsed = diversity_penalty(representation.h)
        result["diversity"] = F.relu(
            torch.as_tensor(diversity_margin, dtype=distance.dtype) - distance
        )
        result["diversity_distance"] = distance.detach()
        result["collapsed_fraction"] = collapsed.detach()
        result["counts"]["diversity"] = batch
    else:
        result["diversity"] = None
        result["counts"]["diversity"] = 0
        result["skipped"]["diversity"] = (
            "single-solution condition (K equals the declared value count); a unique "
            "answer is not forced to split"
        )
    return result


# ----------------------------------------------------------------------------- loop


def _frame(batch: StepBatch, step: int) -> MeasurementFrame:
    return MeasurementFrame(
        values=batch.values[step],
        valid_mask=batch.masks[step],
        frame_ids=(step,),
        episode_ids=(0,),
        camera_id=batch.camera.camera_id,
        sample_xy=batch.sample_xy,
        metadata={"variance": 0.01**2, "variance_is_estimate": False}
        if batch.source == "synthetic"
        else {},
        metadata_available={"variance": batch.source == "synthetic"},
    )


def _percentiles(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"p50": float("nan"), "p95": float("nan"), "n": 0}
    array = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "n": int(array.size),
    }


def _git_sha() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (Linux reports KiB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) * 1024


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _rng_state(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "numpy_bit_generator": rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any], rng: np.random.Generator) -> None:
    rng.bit_generator.state = state["numpy_bit_generator"]
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])


def readout_latency(config: N02Config, representation: Any) -> dict[str, Any]:
    """Latency of the declared readouts on a real representation (measured, not modelled)."""
    readout = TokenReadout(representation.h.shape[3], representation.K, token_dim=config.token_dim)
    budget = min(config.tokens_per_camera, config.K * 16)
    budget -= budget % max(config.K, 1)
    budget = max(budget, config.K)
    timings: dict[str, Any] = {}
    with torch.no_grad():
        for name, function in (("map", map_hypothesis), ("tokens", hypothesis_tokens)):
            samples: list[float] = []
            for _ in range(5):
                start = time.perf_counter()
                function(representation, budget, readout=readout)
                samples.append((time.perf_counter() - start) * 1000.0)
            timings[name] = _percentiles(samples)
    timings["token_budget"] = budget
    return timings


def baseline_latency(
    config: N02Config, batch: StepBatch, *, device: str, threads: int
) -> dict[str, Any]:
    """Single-hypothesis (K=1) latency on the same frames: the nearest baseline."""
    baseline_config = replace(config, K=1)
    baseline_config.measurement = replace(config.measurement)
    model = build_encoder(baseline_config).to(device).eval()
    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    state = model.init_state(batch.batch_size, batch.camera, device)
    samples: list[float] = []
    with torch.no_grad():
        for step in range(min(3, batch.num_steps)):
            start = time.perf_counter()
            output = model.step(_frame(batch, step), EmbodimentContext(), state, budget)
            samples.append((time.perf_counter() - start) * 1000.0)
            state = model.detach_state(output.state)
    del threads
    return _percentiles(samples)


def analytic_reference_latency(config: N02Config) -> dict[str, Any] | None:
    """Latency of the exact two-hypothesis reference update, when it is available."""
    analytic, reason = _analytic_module()
    if analytic is None:
        return None
    case_config = analytic.make_config(K=2, cycles=config.cycles, history=config.history)
    case = analytic.delayed_evidence_case(case_config, delay=3, seed=0)
    samples: list[float] = []
    for _ in range(5):
        start = time.perf_counter()
        buffer = analytic.build_buffer(case)
        analytic.analytic_posterior(case, buffer, case_config)
        samples.append((time.perf_counter() - start) * 1000.0)
    del reason
    return _percentiles(samples)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(args.threads)))
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_payload: dict[str, Any] | None = None
    started_at_step = 0
    resumed_shape = False
    if args.resume:
        checkpoint_payload = load_checkpoint(Path(args.resume))
        config = N02Config.from_dict(checkpoint_payload["config"])
        seed = int(checkpoint_payload["seed"])
        source = str(checkpoint_payload["source"])
        started_at_step = int(checkpoint_payload["step"])
        # The training shape is part of the run, not of the model: a resume that
        # silently fell back to the CLI defaults would draw different batches and
        # could not reproduce the trajectory. Restore it from the checkpoint.
        shape = checkpoint_payload.get("run") or {}
        if shape:
            resumed_shape = True
            args.batch_size = int(shape.get("batch_size", args.batch_size))
            args.sequence_length = int(shape.get("sequence_length", args.sequence_length))
            args.frame_size = int(shape.get("frame_size", args.frame_size))
            args.grad_clip = float(shape.get("grad_clip", args.grad_clip))
            args.diversity_margin = float(shape.get("diversity_margin", args.diversity_margin))
    else:
        config = load_config(args.config, args.preset)
        seed = int(args.seed)
        source = args.data_source
        if source == "auto":
            synthetic, reason = _synthetic_module()
            source = "synthetic" if synthetic is not None else "analytic"
            if synthetic is None:
                print(f"[n02-train] synthetic unavailable ({reason}); using the analytic fallback")
    if source == "analytic":
        # The analytic reference has exactly one measurement channel; the declared
        # head must match it. The override is recorded in summary.json.
        config.measurement.input_mode = "bayer_cfa"
    # Validate the data source's shape requirements before any training work happens.
    if source == "synthetic":
        validate_synthetic_frame_size(int(args.frame_size), seed=seed, split="train")

    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    from owac.baselines.n02 import build_baseline

    arm = (
        checkpoint_payload.get("arm", "candidate")
        if checkpoint_payload
        else getattr(args, "arm", "candidate")
    )
    model = (build_encoder(config) if arm == "candidate" else build_baseline(arm, config)).to(
        device
    )
    config = model.config
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    weights = config.loss_weights
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model"])
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
        _restore_rng(checkpoint_payload["rng"], rng)
        if "particle_rng" in checkpoint_payload:
            model._generator.set_state(checkpoint_payload["particle_rng"])

    budget = ComputeBudget(max_updates=config.cycles, max_history=config.history)
    diversity_active = False
    step = started_at_step
    losses: list[float] = []
    terms_log: dict[str, list[float]] = {}
    encoder_times: list[float] = []
    step_times: list[float] = []
    log_path = output / "train_log.jsonl"
    log_mode = "a" if (args.resume and log_path.exists()) else "w"
    started = time.perf_counter()
    started_cpu = time.process_time()
    load_start = os.getloadavg()
    notes: list[str] = []
    with log_path.open(log_mode) as handle:
        while step < int(args.steps):
            batch = build_batch(
                config,
                rng,
                source=source,
                batch_size=int(args.batch_size),
                num_frames=int(args.sequence_length),
                frame_size=int(args.frame_size),
                seed=seed,
                split="train",
            )
            diversity = diversity_active or batch.multi_solution
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulators = {
                name: TermAccumulator()
                for name in ("measurement", "geometry_set", "revision", "diversity", "mixture")
            }
            step_tensors: dict[str, list[torch.Tensor]] = {name: [] for name in accumulators}
            state = model.init_state(batch.batch_size, batch.camera, device)
            step_start = time.perf_counter()
            step_cpu_start = time.process_time()
            last_output: Any = None
            for index in range(batch.num_steps):
                frame = _frame(batch, index)
                forward_start = time.perf_counter()
                context = batch.contexts[index] if batch.contexts else EmbodimentContext()
                step_output = model.step(frame, context, state, budget)
                encoder_times.append((time.perf_counter() - forward_start) * 1000.0)
                state = model.detach_state(step_output.state)
                representation = representation_from_output(step_output)
                terms = step_terms(
                    representation,
                    anchor=batch.anchor[index],
                    wrong=batch.wrong[index],
                    post_contradiction=batch.post_contradiction[index],
                    observations=int(step_output.diagnostics.get("likelihood_observations", 1)),
                    diversity_active=diversity
                    if batch.identifiable_now is None
                    else not bool(batch.identifiable_now[index].all()),
                    diversity_margin=float(args.diversity_margin),
                    ambiguity_prior=context.hypothesis_prior,
                    identifiable_now=None
                    if batch.identifiable_now is None
                    else batch.identifiable_now[index],
                )
                for name, accumulator in accumulators.items():
                    value = terms[name]
                    if value is None:
                        accumulator.skip(terms["skipped"].get(name, "not defined"))
                    else:
                        accumulator.add(float(value.detach()), terms["counts"][name])
                        step_tensors[name].append(value)
                last_output = step_output
            # The weighted total is rebuilt from the same per-step tensors the
            # reported means come from, so the logged terms and the optimised
            # objective cannot drift apart.
            total = torch.zeros((), device=device)
            for name, values in step_tensors.items():
                if not values:
                    continue
                total = total + float(getattr(weights, name)) * torch.stack(values).mean()
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            step_seconds = time.perf_counter() - step_start
            step_cpu_seconds = time.process_time() - step_cpu_start
            step_times.append(step_seconds)
            loss_value = float(total.detach())
            losses.append(loss_value)
            record: dict[str, Any] = {
                "step": step,
                "loss": loss_value,
                "terms": {name: acc.mean for name, acc in accumulators.items()},
                "weighted": {
                    name: (None if acc.mean is None else float(getattr(weights, name)) * acc.mean)
                    for name, acc in accumulators.items()
                },
                "effective_samples": {name: acc.samples for name, acc in accumulators.items()},
                "active_steps": {name: acc.active_steps for name, acc in accumulators.items()},
                "skipped": {name: acc.skipped for name, acc in accumulators.items() if acc.skipped},
                "diversity_active": diversity,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm),
                "encoder_ms": float(np.mean(encoder_times[-batch.num_steps :])),
                "step_seconds": step_seconds,
                "step_cpu_seconds": step_cpu_seconds,
                "observation_history": int(last_output.diagnostics["observation_history"]),
                "cycles_run": int(last_output.diagnostics["cycles_run"]),
                "state_bytes": int(last_output.cost.state_bytes or 0),
                "top_weight": float(last_output.representation.weights.detach().max()),
                "stop_reason": str(last_output.representation.diagnostics["stop_reason"]),
            }
            for name, accumulator in accumulators.items():
                if accumulator.mean is not None:
                    terms_log.setdefault(name, []).append(accumulator.mean)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            step += 1
            save_checkpoint(
                output / "checkpoint.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.as_dict(),
                    "seed": seed,
                    "step": step,
                    "source": source,
                    "arm": arm,
                    **(
                        {"particle_rng": model._generator.get_state()}
                        if hasattr(model, "_generator")
                        else {}
                    ),
                    "rng": _rng_state(rng),
                    "torch_version": torch.__version__,
                    "run": {
                        "batch_size": int(args.batch_size),
                        "sequence_length": int(args.sequence_length),
                        "frame_size": int(args.frame_size),
                        "grad_clip": float(args.grad_clip),
                        "diversity_margin": float(args.diversity_margin),
                        "lr": float(args.lr),
                        "data_source": source,
                    },
                },
            )
            if step % max(1, int(args.log_every)) == 0 or step == int(args.steps):
                shown = {
                    name: (None if value is None else round(value, 4))
                    for name, value in record["terms"].items()
                }
                print(
                    f"[n02-train] step {step}/{args.steps} loss={loss_value:.4f} "
                    f"terms={shown} step_s={step_seconds:.2f}"
                )
    wall = time.perf_counter() - started
    cpu = time.process_time() - started_cpu
    load_end = os.getloadavg()
    if not losses:
        notes.append("no step was run: --steps is not greater than the resumed checkpoint step")
    with torch.no_grad():
        model.eval()
        representation = representation_from_output(last_output)
        latency = {
            "encoder_step_ms": _percentiles(encoder_times),
            "readout_ms": readout_latency(config, representation),
            "single_hypothesis_baseline_ms": baseline_latency(
                config, batch, device=str(device), threads=int(args.threads)
            ),
            "analytic_reference_ms": analytic_reference_latency(config),
        }
    summary = {
        "command": "python -m owac.training.n02",
        "config_path": None if args.config is None else str(Path(args.config)),
        "preset": args.preset,
        "config": config.as_dict(),
        "seed": seed,
        "git_sha": _git_sha(),
        "source": source,
        "arm": arm,
        "resumed_from": None if args.resume is None else str(Path(args.resume)),
        "first_step": started_at_step,
        "steps_run": len(losses),
        "last_step": step,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "frame_size": int(args.frame_size),
        "lr": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "steps_per_second": (len(losses) / wall) if wall > 0 and losses else None,
        "cpu_seconds_per_step": (cpu / len(losses)) if losses else None,
        "fraction_of_wall_that_is_cpu": (cpu / wall) if wall > 0 else None,
        "load_average_start": list(load_start),
        "load_average_end": list(load_end),
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "best": min(losses) if losses else None,
            "mean_first_5": float(np.mean(losses[:5])) if losses else None,
            "mean_last_5": float(np.mean(losses[-5:])) if losses else None,
            "terms_last": {name: values[-1] for name, values in terms_log.items() if values},
        },
        "peak_rss_bytes": peak_rss_bytes(),
        "parameters": parameter_counts(model),
        "latency_ms": latency,
        "device": str(device),
        "torch_threads": int(torch.get_num_threads()),
        "torch_version": torch.__version__,
        "resumed_run_shape": resumed_shape,
        "notes": notes
        + [
            "measurement loss is nats per current likelihood; geometry is a binary proxy",
            "the hidden geometry is used in the loss only and is never an encoder input",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the N02 measurement-constrained recurrent completion field."
    )
    parser.add_argument("--config", type=Path, default=None, help="N02 JSON config")
    parser.add_argument(
        "--preset", choices=("tiny", "small", "medium"), default=None, help="named scale preset"
    )
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic generator")
    parser.add_argument(
        "--data-source",
        choices=("auto", "synthetic", "analytic"),
        default="auto",
        help="measurement source; 'auto' prefers the synthetic generator",
    )
    parser.add_argument("--steps", type=int, default=200, help="absolute total step count")
    parser.add_argument(
        "--arm",
        default="candidate",
        choices=(
            "candidate",
            "single_hypothesis",
            "plain_recurrent",
            "particle_filter",
            "dense_global_recurrent",
            "no_dependency_propagation",
            "no_contradiction_reset",
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/n02-smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="frames per training sequence; the cost is linear in it",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=64,
        help=(
            "synthetic frame side (even). 64 is the default because it is the cheapest "
            "side whose mechanism grid is still 16x16; the generator's own default is 80, "
            "which is ~7x slower per training step at batch 4. Illegal sides are refused "
            "before training starts, with the generator's requirement in the message."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to restore")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    parser.add_argument("--diversity-margin", type=float, default=DIVERSITY_MARGIN)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.synthetic:
        args.data_source = "synthetic"
    if args.config is None and args.preset is None:
        args.preset = "tiny"
    summary = train(args)
    print(json.dumps({k: summary[k] for k in ("source", "steps_run", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

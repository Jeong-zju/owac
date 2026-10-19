"""N01 v1.0 evaluation entry points.

P1 ships one subcommand, ``analytic``: it runs the seven numeric/behavioural
checks of ``docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md`` section 4 plus the
one-dimensional same-semantics example, prints a compact table and writes the
machine-readable receipt.  Every case reports the expected value, the observed
value, the tolerance it was judged with and the numeric condition numbers, so a
later phase can re-run the same commands without re-deriving the numbers.

The suite exercises both the production modules (``ledger``, ``domains``,
``statistics``, ``replay``) and the independent NumPy FP64 oracle
(``reference``); the oracle is never given production results.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full import reference as ref
from owac.frameworks.n01_full import replay as replay_module
from owac.frameworks.n01_full.config import ReplayConfig
from owac.frameworks.n01_full.contracts import DependencyNode, SourceKey
from owac.frameworks.n01_full.domains import (
    VersionedDomain,
    make_domain,
    merge_domains,
    restore_sources,
    retract_sources,
    split_domain,
)
from owac.frameworks.n01_full.ledger import ContentReplacement, MeasurementLedger
from owac.frameworks.n01_full.sequence1d import sequence1d_example
from owac.frameworks.n01_full.statistics import (
    GaussianPrior,
    InterpolationSpec,
    NoiseModel,
    fit_structure,
)

FP64_ATOL = 1e-10
FP64_RTOL = 1e-8
FP32_ATOL = 1e-5
FP32_RTOL = 1e-4

DATASET = "analytic"


@dataclass(frozen=True)
class CaseResult:
    """One check: expected, observed, tolerance and the raw numbers behind it."""

    name: str
    passed: bool
    expected: dict[str, Any]
    observed: dict[str, Any]
    tolerance: dict[str, float]
    conditions: dict[str, Any]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": "pass" if self.passed else "fail",
            "expected": self.expected,
            "observed": self.observed,
            "tolerance": self.tolerance,
            "conditions": self.conditions,
            "notes": self.notes,
        }


def _close(
    observed: float, expected: float, *, atol: float = FP64_ATOL, rtol: float = FP64_RTOL
) -> bool:
    return bool(abs(observed - expected) <= atol + rtol * abs(expected))


def _frame(
    values: Any,
    *,
    frame_id: int | str = 0,
    episode_id: str = "e0",
    camera_id: str = "c0",
    valid: Any | None = None,
    bad: Any | None = None,
    saturated: Any | None = None,
    new_measurement: bool = True,
    timestamp: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> MeasurementFrame:
    """A one-camera frame; a 1-D list of length ``L`` becomes a ``1 x 1 x L`` grid."""
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, 1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.reshape(1, 1, *tensor.shape)
    elif tensor.ndim == 3:
        tensor = tensor[None]
    return MeasurementFrame(
        values=tensor,
        valid_mask=None if valid is None else torch.as_tensor(valid).reshape(tensor.shape),
        bad_mask=None if bad is None else torch.as_tensor(bad).reshape(tensor.shape),
        saturated_mask=(
            None if saturated is None else torch.as_tensor(saturated).reshape(tensor.shape)
        ),
        episode_ids=[episode_id],
        frame_ids=[frame_id],
        camera_id=camera_id,
        timestamp=torch.tensor([float(timestamp)]),
        new_measurement=torch.tensor([new_measurement]),
        metadata=dict(metadata or {}),
    )


def _key(
    frame_id: int | str,
    position: int,
    *,
    episode_id: str = "e0",
    camera_id: str = "c0",
    channel: int = 0,
) -> SourceKey:
    return SourceKey(DATASET, episode_id, camera_id, frame_id, channel, 0, position)


def _ledger_with(values: Any, **kwargs: Any) -> MeasurementLedger:
    ledger = MeasurementLedger(DATASET, value_units="analytic_unit")
    ledger.observe(_frame(values, **kwargs))
    return ledger


def _variance_model(
    keys: Sequence[SourceKey], variances: Sequence[float], **kwargs: Any
) -> NoiseModel:
    return NoiseModel.diagonal(dict(zip(keys, variances, strict=True)), **kwargs)


def _reference_fit(
    ledger: MeasurementLedger,
    domain: VersionedDomain,
    *,
    dimension: int = 2,
    basis: str = "affine",
    variances: dict[SourceKey, float] | None = None,
) -> ref.ReferenceFit:
    """Build the oracle's fit from raw records only; production output is never passed in."""
    observations = []
    for key in domain.evidence_source_ids:
        if not ledger.has_source(key):
            continue
        record = ledger.get_source(key)
        if record.validity_kind != "valid":
            continue
        x, y = ledger.sample_coordinates(key)
        variance = 1.0 if variances is None else float(variances[key])
        observations.append(
            ref.ReferenceObservation(
                source_id=key,
                phase=key.channel_or_phase,
                xy=(x, y),
                value=float(record.raw_value),
                variance=variance,
            )
        )
    return ref.reference_fit(observations, basis=basis, dimension=dimension)


# ------------------------------------------------------------------------------
# case 1: duplicate ids do not become independent observations
# ------------------------------------------------------------------------------


def case_duplicate_then_new_id() -> CaseResult:
    ledger = _ledger_with([1.0, 3.0])
    keys = [_key(0, 0), _key(0, 1)]
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    first = fit_structure(domain, ledger, None, _variance_model(keys, [1.0, 1.0]))
    # Re-send the same packet: the same SourceKeys, a renewed arrival event.
    duplicate = ledger.observe(_frame([1.0, 3.0], frame_id=0))
    replay = fit_structure(domain, ledger, None, _variance_model(keys, [1.0, 1.0]))
    # A genuinely new exposure with an identical *value* is still a new measurement.
    ledger.observe(_frame([1.0], frame_id=9))
    identical_value_key = _key(9, 0)
    third = fit_structure(
        make_domain(
            "D3",
            reference_chart="ref",
            basis="phase_local_constant",
            phase=0,
            evidence_source_ids=[*keys, identical_value_key],
            decision_version="dv1",
            mapping_version="mv1",
        ),
        ledger,
        None,
        _variance_model([*keys, identical_value_key], [1.0, 1.0, 1.0]),
    )
    mean = float(first.estimate.theta[0])
    variance = float(first.estimate.covariance[0, 0])
    replay_mean = float(replay.estimate.theta[0])
    replay_variance = float(replay.estimate.covariance[0, 0])
    third_mean = float(third.estimate.theta[0])
    third_variance = float(third.estimate.covariance[0, 0])
    passed = (
        _close(mean, 2.0)
        and _close(variance, 0.5)
        and _close(replay_mean, 2.0)
        and _close(replay_variance, 0.5)
        and duplicate.new_samples == 0
        and duplicate.duplicate_frames
        and _close(third_mean, 5.0 / 3.0)
        and _close(third_variance, 1.0 / 3.0)
        and third.estimate.independent_observation_count == 3
    )
    return CaseResult(
        name="duplicate_then_new_id",
        passed=passed,
        expected={
            "mean_two_observations": 2.0,
            "variance_two_observations": 0.5,
            "variance_after_replay": 0.5,
            "mean_identical_value_new_exposure": 5.0 / 3.0,
            "variance_three_observations": 1.0 / 3.0,
            "new_samples_from_duplicate": 0,
        },
        observed={
            "mean": mean,
            "variance": variance,
            "replay_mean": replay_mean,
            "replay_variance": replay_variance,
            "duplicate_new_samples": duplicate.new_samples,
            "third_mean": third_mean,
            "third_variance": third_variance,
            "third_observation_count": third.estimate.independent_observation_count,
            "unique_frames": ledger.unique_frame_count(camera_id="c0"),
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={"condition_number": first.condition_number},
        notes=(
            "a repeated packet keeps one identity; an identical value in a new exposure "
            "is a new measurement and is counted as one"
        ),
    )


# ------------------------------------------------------------------------------
# case 2: correlated observations and the precision-slice counterexample
# ------------------------------------------------------------------------------


def case_correlated_two_observation() -> CaseResult:
    ledger = _ledger_with([1.0, 3.0])
    keys = [_key(0, 0), _key(0, 1)]
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    noise = NoiseModel.dense(keys, [[1.0, 0.5], [0.5, 1.0]])
    fit = fit_structure(domain, ledger, None, noise)
    retracted = fit_structure(retract_sources(domain, [keys[1]]), ledger, None, noise)
    mean = float(fit.estimate.theta[0])
    variance = float(fit.estimate.covariance[0, 0])
    mean_after = float(retracted.estimate.theta[0])
    variance_after = float(retracted.estimate.covariance[0, 0])
    # The documented wrong version: slice the old precision matrix and invert it.
    old_precision = torch.linalg.inv(noise.matrix)
    wrong_precision_slice = float(1.0 / old_precision[0, 0])
    passed = (
        _close(mean, 2.0)
        and _close(variance, 0.75)
        and _close(mean_after, 1.0)
        and _close(variance_after, 1.0)
        and not _close(wrong_precision_slice, variance_after)
    )
    return CaseResult(
        name="correlated_two_observation",
        passed=passed,
        expected={
            "mean": 2.0,
            "variance_from_Sigma": 0.75,
            "variance_after_retracting_second": 1.0,
            "precision_slice_is_wrong": True,
        },
        observed={
            "mean": mean,
            "variance": variance,
            "mean_after_retract": mean_after,
            "variance_after_retract": variance_after,
            "wrong_precision_slice_variance": wrong_precision_slice,
            "wrong_version_detected": not _close(wrong_precision_slice, variance_after),
            "observation_count": fit.estimate.independent_observation_count,
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={"condition_number": fit.condition_number},
        notes=(
            "retraction takes the remaining sub-block of the observation covariance and "
            "re-solves; slicing the old precision gives 0.75 and is rejected"
        ),
    )


# ------------------------------------------------------------------------------
# case 3: affine field, fractional coordinates, heteroscedastic noise, two phases
# ------------------------------------------------------------------------------


def case_affine_heteroscedastic() -> CaseResult:
    generator = torch.Generator().manual_seed(7)
    height, width = 3, 4
    field = torch.zeros((2, height, width), dtype=torch.float64)
    coordinates = torch.stack(
        torch.meshgrid(
            torch.arange(width, dtype=torch.float64),
            torch.arange(height, dtype=torch.float64),
            indexing="xy",
        ),
        dim=-1,
    )
    for phase in range(2):
        field[phase] = (
            0.5 * phase + 1.25 * coordinates[..., 0] - 0.75 * coordinates[..., 1] + 0.1 * phase
        )
    noise_scale = torch.linspace(0.05, 0.2, height * width, dtype=torch.float64).reshape(
        height, width
    )
    values = field + noise_scale.reshape(height, width) * torch.randn(
        (2, height, width), generator=generator
    )
    # Fractional sampling coordinates: declared offsets in (x, y), never a fabricated grid.
    offsets = (coordinates + 0.25 * torch.ones_like(coordinates)).unsqueeze(0).repeat(2, 1, 1, 1)
    ledger = MeasurementLedger(DATASET, value_units="analytic_unit")
    ledger.observe(
        MeasurementFrame(
            values=values.reshape(1, 2, height, width),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
            sample_xy=offsets.reshape(1, 2, height, width, 2),
        )
    )
    keys = [
        SourceKey(DATASET, "e0", "c0", 0, phase, y, x)
        for phase in range(2)
        for y in range(height)
        for x in range(width)
    ]
    variances = {key: float(noise_scale[key.sensor_y, key.sensor_x] ** 2) for key in keys}
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_affine",
        phase=None,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    fit = fit_structure(domain, ledger, None, NoiseModel.diagonal(variances))
    oracle = _reference_fit(ledger, domain, basis="affine", dimension=2, variances=variances)
    queries = [(0, 1.0, 0.5), (0, 2.5, 1.5), (1, 0.5, 2.0), (1, 3.0, 0.0)]
    production_means, production_variances = fit.predictions(queries)
    worst_mean, worst_variance = 0.0, 0.0
    for index, (phase, x, y) in enumerate(queries):
        oracle_mean, oracle_variance = oracle.predict(phase, (x, y))
        worst_mean = max(worst_mean, abs(float(production_means[index]) - oracle_mean))
        # The oracle has no declared noise model beyond the diagonal variances, so the
        # comparison is against the same diagonal model, not against production code.
        worst_variance = max(
            worst_variance, abs(float(production_variances[index]) - oracle_variance)
        )
    passed = (
        worst_mean <= FP64_ATOL + FP64_RTOL * 10.0
        and worst_variance <= FP64_ATOL + FP64_RTOL * 10.0
        and set(fit.blocks) == {0, 1}
    )
    return CaseResult(
        name="affine_heteroscedastic",
        passed=passed,
        expected={
            "production_matches_oracle_mean": True,
            "production_matches_oracle_variance": True,
            "phases_fitted_separately": True,
        },
        observed={
            "max_abs_mean_difference": worst_mean,
            "max_abs_variance_difference": worst_variance,
            "phases": sorted(str(item) for item in fit.blocks),
            "condition_numbers": {
                str(phase): fit.blocks[phase].condition_number for phase in fit.blocks
            },
            "queries": [list(item) for item in queries],
            "production_means": [float(item) for item in production_means],
            "oracle_means": [oracle.predict(phase, (x, y))[0] for phase, x, y in queries],
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={"seed": 7, "grid": [height, width], "phases": 2},
        notes=(
            "fractional sample coordinates and per-sample variances; both phases fitted separately"
        ),
    )


# ------------------------------------------------------------------------------
# case 4: set algebra under fixed versions
# ------------------------------------------------------------------------------


def case_set_algebra() -> CaseResult:
    first_two = [_key(0, 0), _key(0, 1)]
    second_two = [_key(0, 2), _key(0, 3)]
    left = make_domain(
        "A",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=first_two,
        decision_version="dv1",
        mapping_version="mv1",
    )
    right = make_domain(
        "B",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=[first_two[1], second_two[0]],
        decision_version="dv1",
        mapping_version="mv1",
    )
    merge_ab = merge_domains([left, right])
    merge_ba = merge_domains([right, left])
    children = split_domain(
        merge_ab,
        [("A1", [first_two[0], second_two[0]]), ("A2", [first_two[1]])],
    )
    remerged = merge_domains(list(children), domain_id="A+B")
    retracted = retract_sources(merge_ab, [first_two[1]])
    restored = restore_sources(retracted, [first_two[1]])
    key_set = lambda domain: {key.identity() for key in domain.evidence_source_ids}  # noqa: E731
    # The same events consumed in two chunks must give the same ledger and fit.
    chunked = MeasurementLedger(DATASET, value_units="analytic_unit")
    chunked.observe(_frame([1.0, 3.0], frame_id=0))
    chunked.observe(_frame([2.0, 4.0], frame_id=1))
    single = MeasurementLedger(DATASET, value_units="analytic_unit")
    single.observe(
        MeasurementFrame(
            values=torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.float64).reshape(1, 1, 1, 4),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
        )
    )
    chunked_keys = [_key(0, 0), _key(0, 1), _key(1, 0), _key(1, 1)]
    single_keys = [_key(0, index) for index in range(4)]
    chunked_values = [chunked.get_source(key).raw_value for key in chunked_keys]
    single_values = [single.get_source(key).raw_value for key in single_keys]
    version_changed = make_domain(
        "A",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=first_two,
        decision_version="dv2",
        mapping_version="mv1",
    )
    passed = (
        merge_ab.content_digest == merge_ba.content_digest
        and merge_ab.support_version == merge_ba.support_version
        and key_set(merge_ab) == key_set(left) | key_set(right)
        and len(merge_ab.evidence_source_ids) == 3
        and remerged.content_digest == merge_ab.content_digest
        and restored.content_digest == merge_ab.content_digest
        and restored.support_version != merge_ab.support_version
        and key_set(retracted) == key_set(merge_ab) - {first_two[1].identity()}
        and chunked_values == single_values
        and version_changed.content_digest != left.content_digest
    )
    return CaseResult(
        name="set_algebra",
        passed=passed,
        expected={
            "merge_is_order_independent": True,
            "merge_deduplicates_shared_sources": 3,
            "split_then_merge_restores_content": True,
            "retract_then_restore_restores_content": True,
            "chunked_events_match_single_frame": True,
            "declared_version_change_changes_digest": True,
        },
        observed={
            "merge_versions_equal": merge_ab.support_version == merge_ba.support_version,
            "merged_evidence_count": len(merge_ab.evidence_source_ids),
            "split_then_merge_digest_equal": remerged.content_digest == merge_ab.content_digest,
            "restore_digest_equal": restored.content_digest == merge_ab.content_digest,
            "restore_version_differs": restored.support_version != merge_ab.support_version,
            "retracted_members": len(retracted.evidence_source_ids),
            "chunked_values": chunked_values,
            "single_values": single_values,
            "version_changed": version_changed.content_digest != left.content_digest,
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={
            "merge_ab_version": merge_ab.support_version,
            "merge_ba_version": merge_ba.support_version,
            "restored_version": restored.support_version,
        },
        notes=(
            "versions are only compared for fixed declarations; lineage is kept separate "
            "from content"
        ),
    )


# ------------------------------------------------------------------------------
# case 5: missing/bad/saturated are not zero observations; rank deficiency is labelled
# ------------------------------------------------------------------------------


def case_validity_semantics() -> CaseResult:
    values = [10.0, 1.0, 999.0, 3.0, 55.0]
    valid = [False, True, False, True, True]
    bad = [False, False, True, False, False]
    saturated = [False, False, False, False, True]
    ledger = _ledger_with(values, valid=valid, bad=bad, saturated=saturated)
    all_keys = [_key(0, position) for position in range(5)]
    usable = [all_keys[1], all_keys[3]]
    domain_all = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=all_keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    fit = fit_structure(domain_all, ledger, None, _variance_model(all_keys, [1.0] * 5))
    domain_usable = make_domain(
        "U",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=usable,
        decision_version="dv1",
        mapping_version="mv1",
    )
    fit_usable = fit_structure(domain_usable, ledger, None, _variance_model(usable, [1.0, 1.0]))
    empty = make_domain(
        "E",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=[all_keys[0]],
        decision_version="dv1",
        mapping_version="mv1",
    )
    ledger_missing = _ledger_with([1.0], valid=[False])
    fit_empty = fit_structure(empty, ledger_missing, None, _variance_model([all_keys[0]], [1.0]))
    # Rank deficiency: the affine basis cannot identify the y slope from two
    # samples that share x, and the mask must say which coordinates remain.
    rank_ledger = MeasurementLedger(DATASET, value_units="analytic_unit")
    rank_ledger.observe(_frame([[1.0, 2.0], [1.0, 2.0]], frame_id=0))
    rank_keys = [SourceKey(DATASET, "e0", "c0", 0, 0, y, 0) for y in range(2)]
    rank_domain = make_domain(
        "R",
        reference_chart="ref",
        basis="phase_local_affine",
        phase=0,
        evidence_source_ids=rank_keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    fit_rank = fit_structure(rank_domain, rank_ledger, None, _variance_model(rank_keys, [1.0, 1.0]))
    passed = (
        fit.estimate.independent_observation_count == 2
        and {reason for _, reason in fit.excluded_sources} == {"missing", "bad", "saturated"}
        and _close(float(fit.estimate.theta[0]), 2.0)
        and _close(float(fit_usable.estimate.theta[0]), 2.0)
        and fit_empty.rank == 0
        and bool((~fit_empty.estimate.identifiable_mask).all())
        and fit_empty.estimate.uncertainty_kind == "none"
        and fit_empty.estimate.theta is None
        and fit_rank.rank == 2
        and fit_rank.estimate.uncertainty_kind == "none"
        and fit_rank.estimate.covariance is None
        and bool(fit_rank.estimate.identifiable_mask[0])
        and not bool(fit_rank.estimate.identifiable_mask[1])
    )
    return CaseResult(
        name="validity_semantics",
        passed=passed,
        expected={
            "used_observations": 2,
            "excluded_reasons": ["missing", "bad", "saturated"],
            "mean_from_usable_only": 2.0,
            "all_invalid_mask": "all false",
            "rank_deficient_mask": [True, False, True],
            "rank_deficient_uncertainty": "none",
        },
        observed={
            "used_observations": fit.estimate.independent_observation_count,
            "excluded": [
                {"source": list(key.identity()), "reason": reason}
                for key, reason in fit.excluded_sources
            ],
            "mean": float(fit.estimate.theta[0]),
            "empty_rank": fit_empty.rank,
            "empty_mask": [bool(item) for item in fit_empty.estimate.identifiable_mask],
            "empty_uncertainty": fit_empty.estimate.uncertainty_kind,
            "rank_deficient_rank": fit_rank.rank,
            "rank_deficient_mask": [bool(item) for item in fit_rank.estimate.identifiable_mask],
            "rank_deficient_uncertainty": fit_rank.estimate.uncertainty_kind,
            "rank_deficient_covariance": fit_rank.estimate.covariance is None,
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={
            "rank_deficient_condition_number": fit_rank.condition_number,
            "rank_deficient_leverage": fit_rank.diagnostics.get("leverage"),
        },
        notes=(
            "bad and saturated win over valid; a missing sample carries no value; a "
            "rank-deficient design reports the identifiable subspace instead of fake precision"
        ),
    )


# ------------------------------------------------------------------------------
# case 6: cameras and episodes are independent; partial reset is partial
# ------------------------------------------------------------------------------


def case_camera_episode_isolation() -> CaseResult:
    values = torch.tensor([[[[1.0, 3.0]], [[10.0, 30.0]]]], dtype=torch.float64)  # [1,2,1,2]
    ledger = MeasurementLedger(DATASET, value_units="analytic_unit")
    ledger.observe(
        MeasurementFrame(
            values=values,
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
        )
    )
    ledger.observe(
        MeasurementFrame(
            values=torch.tensor([[[[10.0, 30.0]]]], dtype=torch.float64),
            episode_ids=["e0"],
            frame_ids=[0],
            camera_id="c1",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
        )
    )
    ledger.observe(
        MeasurementFrame(
            values=values,
            episode_ids=["e1"],
            frame_ids=[0],
            camera_id="c0",
            timestamp=torch.tensor([0.0]),
            new_measurement=torch.tensor([True]),
        )
    )
    # One batch with two episodes: resetting the first must not touch the second.
    reset_ledger = MeasurementLedger(DATASET, value_units="analytic_unit")
    batch = MeasurementFrame(
        values=torch.tensor([[[[1.0, 3.0]]], [[[7.0, 9.0]]]], dtype=torch.float64),
        episode_ids=["b0", "b1"],
        frame_ids=[0, 0],
        camera_id="c0",
        timestamp=torch.tensor([0.0, 0.0]),
        new_measurement=torch.tensor([True, True]),
    )
    reset_ledger.observe(batch)
    before = reset_ledger.get_source(_key(0, 0, episode_id="b1")).raw_value
    reset_ledger.reset(["b0"])
    after = reset_ledger.get_source(_key(0, 0, episode_id="b1")).raw_value
    c0_key = _key(0, 0, camera_id="c0")
    c1_key = _key(0, 0, camera_id="c1")
    e1_key = _key(0, 0, episode_id="e1")
    c0_values = [ledger.get_source(_key(0, x, camera_id="c0")).raw_value for x in (0, 1)]
    c1_values = [ledger.get_source(_key(0, x, camera_id="c1")).raw_value for x in (0, 1)]
    e1_values = [ledger.get_source(_key(0, x, episode_id="e1")).raw_value for x in (0, 1)]
    passed = (
        c0_values == [1.0, 3.0]
        and c1_values == [10.0, 30.0]
        and e1_values == [1.0, 3.0]
        and ledger.unique_frame_count(camera_id="c0") == 2
        and ledger.unique_frame_count(camera_id="c1") == 1
        and before == 7.0
        and after == 7.0
        and not reset_ledger.has_source(_key(0, 0, episode_id="b0"))
    )
    return CaseResult(
        name="camera_episode_isolation",
        passed=passed,
        expected={
            "c0_values": [1.0, 3.0],
            "c1_values": [10.0, 30.0],
            "e1_values": [1.0, 3.0],
            "c0_frames": 2,
            "c1_frames": 1,
            "batch_second_member_after_partial_reset": 7.0,
        },
        observed={
            "c0_values": c0_values,
            "c1_values": c1_values,
            "e1_values": e1_values,
            "c0_frames": ledger.unique_frame_count(camera_id="c0"),
            "c1_frames": ledger.unique_frame_count(camera_id="c1"),
            "batch_member_before": before,
            "batch_member_after": after,
            "reset_removed_first_member": not reset_ledger.has_source(_key(0, 0, episode_id="b0")),
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={
            "identity_fields": [
                list(c0_key.identity()),
                list(c1_key.identity()),
                list(e1_key.identity()),
            ]
        },
        notes=(
            "identity includes dataset, episode, camera and frame id; equal frame ids never collide"
        ),
    )


# ------------------------------------------------------------------------------
# case 7: both replay interfaces, the exact-window boundary and version staleness
# ------------------------------------------------------------------------------


def _replay_stream(frames: int) -> replay_module.EventStream:
    events = [
        replay_module.ObserveEvent(_frame([float(index + 1), float(index + 2)], frame_id=index))
        for index in range(frames)
    ]
    return replay_module.EventStream(DATASET, tuple(events), value_units="analytic_unit")


def case_replay_interfaces() -> CaseResult:
    stream = _replay_stream(3)
    all_keys = [_key(frame, position) for frame in range(3) for position in (0, 1)]

    def builder(ledger: MeasurementLedger) -> list[VersionedDomain]:
        evidence = [key for key in all_keys if ledger.has_source(key)]
        return [
            make_domain(
                "D",
                reference_chart="ref",
                basis="phase_local_constant",
                phase=0,
                evidence_source_ids=evidence,
                decision_version="dv1",
                mapping_version="mv1",
            )
        ]

    grid = replay_module.ReadoutGrid((replay_module.QueryPoint("q0", "D", 0, 0.0, 0.0),))
    full = replay_module.full_replay(stream, domain_builder=builder, readout_grid=grid)
    target = all_keys[3]
    frozen = builder(full.ledger.without([target]))
    replacement = ContentReplacement("garbage", {target: 1234.0})
    clean = replay_module.fixed_decision_replay(
        stream,
        frozen_domains=frozen,
        retracted_ids=[target],
        replacement=replacement,
        readout_grid=grid,
    )
    leaky = replay_module.fixed_decision_replay(
        stream,
        frozen_domains=builder(full.ledger),
        retracted_ids=[target],
        replacement=replacement,
        readout_grid=grid,
    )

    def reference_fn(
        ledger: MeasurementLedger,
        domains: Sequence[VersionedDomain],
        grid_: replay_module.ReadoutGrid,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for domain in domains:
            oracle = _reference_fit(ledger, domain, basis="constant", dimension=2)
            for point in grid_.points:
                if point.domain_id == domain.domain_id:
                    out[point.label] = oracle.predict(point.phase, (point.x, point.y))[0]
        return out

    counterfactual = replay_module.counterfactual_replay(
        stream,
        domain_builder=builder,
        retracted_ids=[target],
        readout_grid=grid,
        reference_fn=reference_fn,
    )
    # The exact window is bounded: retracting a frame older than the window is a rebuild.
    long_stream = _replay_stream(10)
    long_keys = [_key(frame, 0) for frame in range(10)]

    def long_builder(ledger: MeasurementLedger) -> list[VersionedDomain]:
        evidence = [key for key in long_keys if ledger.has_source(key)]
        return [
            make_domain(
                "D",
                reference_chart="ref",
                basis="phase_local_constant",
                phase=0,
                evidence_source_ids=evidence,
                decision_version="dv1",
                mapping_version="mv1",
            )
        ]

    beyond = replay_module.counterfactual_replay(
        long_stream,
        domain_builder=long_builder,
        retracted_ids=[long_keys[0]],
        replay_config=ReplayConfig(exact_window_unique_frames=8),
        readout_grid=replay_module.ReadoutGrid((replay_module.QueryPoint("q0", "D", 0, 0.0, 0.0),)),
    )
    # Version staleness: a parameter update retires derived nodes (contract rule E4).
    node = DependencyNode(
        node_id="readout",
        kind="toy.readout",
        parent_ids=("output",),
        parameter_version="p1",
        support_version="s1",
        mapping_version="m1",
        calibration_version="c1",
    )
    stale_on_parameter = node.is_stale_against({"parameter_version": "p2"})
    fresh_on_same = not node.is_stale_against({"parameter_version": "p1", "support_version": "s1"})
    passed = (
        clean.unchanged
        and clean.max_abs_diff == 0.0
        and not leaky.unchanged
        and counterfactual.matches_reference is True
        and counterfactual.claim == "exact_independence_in_event_stream"
        and beyond.beyond_exact_range
        and beyond.claim == "rebuild_from_legal_boundary"
        and stale_on_parameter
        and fresh_on_same
    )
    return CaseResult(
        name="replay_interfaces",
        passed=passed,
        expected={
            "fixed_decision_clean_unchanged": True,
            "fixed_decision_leaky_detected": True,
            "counterfactual_matches_independent_reference": True,
            "beyond_exact_window_claim": "rebuild_from_legal_boundary",
            "parameter_update_invalidates_derived_node": True,
        },
        observed={
            "clean_unchanged": clean.unchanged,
            "clean_max_abs_diff": clean.max_abs_diff,
            "leaky_unchanged": leaky.unchanged,
            "leaky_max_abs_diff": leaky.max_abs_diff,
            "counterfactual_matches": counterfactual.matches_reference,
            "counterfactual_max_abs_diff": counterfactual.reference_max_abs_diff,
            "counterfactual_claim": counterfactual.claim,
            "beyond_exact_range": beyond.beyond_exact_range,
            "beyond_claim": beyond.claim,
            "history_loss_frames": len(beyond.history_loss),
            "stale_on_parameter": stale_on_parameter,
            "fresh_on_same_versions": fresh_on_same,
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={
            "full_replay_range": {
                "oldest": full.exact_replay_range.oldest_frame_id,
                "newest": full.exact_replay_range.newest_frame_id,
                "unique_frames": full.exact_replay_range.unique_frames,
            },
            "frozen_domain_versions": list(clean.frozen_domain_versions),
        },
        notes=(
            "content replacement under frozen decisions is falsifiable (the leaky control "
            "differs); only counterfactual replay claims whole-result independence"
        ),
    )


# ------------------------------------------------------------------------------
# case 8: interpolation, prior and the conservative bound
# ------------------------------------------------------------------------------


def case_interpolation_and_bounds() -> CaseResult:
    ledger = _ledger_with([1.0, 3.0, 2.0])
    keys = [_key(0, position) for position in range(3)]
    domain = make_domain(
        "D",
        reference_chart="ref",
        basis="phase_local_constant",
        phase=0,
        evidence_source_ids=keys,
        decision_version="dv1",
        mapping_version="mv1",
    )
    # Two rows pull from the same source pair: duplicate information shows up as a
    # correlated (here singular) row block, which must not be inverted blindly.
    weights = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
    spec = InterpolationSpec(row_positions=((0.0, 0.0), (1.0, 0.0)), weights=weights)
    singular = fit_structure(domain, ledger, spec, _variance_model(keys, [1.0, 1.0, 1.0]))
    fallback = bool(singular.diagnostics.get("singular_interpolation_fallback"))
    # A prior is counted once and never becomes an observation.
    prior = GaussianPrior(
        prior_id="prior-v1",
        mean=torch.tensor([0.0]),
        covariance=torch.tensor([[4.0]]),
    )
    with_prior = fit_structure(
        domain, ledger, None, _variance_model(keys, [1.0, 1.0, 1.0]), prior=prior
    )
    without_prior = fit_structure(domain, ledger, None, _variance_model(keys, [1.0, 1.0, 1.0]))
    # Conservative bound under unknown correlation: sample equicorrelated noise and
    # check the empirical variance stays below the declared bound.
    bound_fit = fit_structure(
        domain,
        ledger,
        None,
        NoiseModel.unknown_correlation(
            {key: 0.5 for key in keys}, version="noise-unknown-correlation-test"
        ),
    )
    generator = torch.Generator().manual_seed(11)
    draws = 4000
    shared = torch.randn(draws, generator=generator) * 0.5
    idiosyncratic = torch.randn((draws, 3), generator=generator) * 0.5
    samples = shared[:, None] + idiosyncratic + torch.tensor([1.0, 3.0, 2.0])
    empirical = float(samples.mean(dim=1).var(unbiased=True))
    bound_matrix = bound_fit.estimate.covariance
    bound = float(bound_matrix[0, 0]) if bound_matrix is not None else float("nan")
    passed = (
        fallback
        and singular.estimate.independent_observation_count == 3
        and with_prior.estimate.prior_id == "prior-v1"
        and with_prior.estimate.independent_observation_count == 3
        and not without_prior.estimate.has_prior
        and bound_fit.estimate.uncertainty_kind == "proxy"
        and empirical <= bound
        and empirical < bound
    )
    return CaseResult(
        name="interpolation_and_bounds",
        passed=passed,
        expected={
            "singular_interpolation_falls_back_to_raw_factors": True,
            "prior_counts_once_not_as_observation": True,
            "unknown_correlation_uses_proxy_label": True,
            "conservative_bound_covers_empirical_variance": True,
        },
        observed={
            "singular_fallback": fallback,
            "interpolation_error": singular.diagnostics.get("singular_interpolation_error"),
            "interpolated_observation_count": singular.estimate.independent_observation_count,
            "prior_id": with_prior.estimate.prior_id,
            "prior_observation_count": with_prior.estimate.independent_observation_count,
            "prior_uncertainty": with_prior.estimate.uncertainty_kind,
            "empirical_variance": empirical,
            "conservative_bound": bound,
            "bound_ratio": bound / empirical if empirical > 0 else None,
            "proxy_uncertainty": bound_fit.estimate.uncertainty_kind,
        },
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={"seed": 11, "draws": draws, "correlation": "equicorrelated 0.5"},
        notes=(
            "a singular interpolation block keeps the raw observations as factors; the "
            "conservative bound is a labelled proxy, never mixed with the exact covariance"
        ),
    )


# ------------------------------------------------------------------------------
# case 9: the one-dimensional same-semantics example
# ------------------------------------------------------------------------------


def case_sequence1d() -> CaseResult:
    example = sequence1d_example(frames=4, length=8, seed=0)
    second = sequence1d_example(frames=3, length=5, seed=3)
    passed = (
        example["duplicate_measurements_added"] == 0
        and example["unique_frames"] == 4
        and example["independent_observation_count"] == 32
        and example["retracted_observation_count"] == 31
        and example["variance_delta"] > 0
        and example["raw_record_untouched"]
        and second["unique_frames"] == 3
    )
    return CaseResult(
        name="sequence1d",
        passed=passed,
        expected={
            "unique_frames_after_duplicate": 4,
            "duplicate_measurements_added": 0,
            "observations": 32,
            "observations_after_retraction": 31,
            "raw_record_untouched": True,
        },
        observed=example,
        tolerance={"atol": FP64_ATOL, "rtol": FP64_RTOL},
        conditions={"second_seed_check": {"seed": 3, "unique_frames": second["unique_frames"]}},
        notes=(
            "same semantics as the old 1-D adapter, re-expressed in the P1 contract: "
            "unique-frame buffer, declared heteroscedastic variance, retraction without "
            "rewriting raw"
        ),
    )


CASES: dict[str, Callable[[], CaseResult]] = {
    "duplicate_then_new_id": case_duplicate_then_new_id,
    "correlated_two_observation": case_correlated_two_observation,
    "affine_heteroscedastic": case_affine_heteroscedastic,
    "set_algebra": case_set_algebra,
    "validity_semantics": case_validity_semantics,
    "camera_episode_isolation": case_camera_episode_isolation,
    "replay_interfaces": case_replay_interfaces,
    "interpolation_and_bounds": case_interpolation_and_bounds,
    "sequence1d": case_sequence1d,
}


def run_analytic(cases: Sequence[str] | None = None) -> dict[str, Any]:
    """Run the analytic suite and return the receipt payload."""
    selected = list(cases) if cases else list(CASES)
    unknown = sorted(set(selected) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown analytic case(s): {unknown}; known: {sorted(CASES)}")
    results = [CASES[name]() for name in selected]
    return {
        "format": "owac.n01_full.analytic.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "tolerances": {
            "fp64": {"atol": FP64_ATOL, "rtol": FP64_RTOL},
            "fp32": {"atol": FP32_ATOL, "rtol": FP32_RTOL},
        },
        "cases": [result.to_dict() for result in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results if result.passed),
            "failed": [result.name for result in results if not result.passed],
        },
    }


def _write_receipt(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "analytic.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (output / "analytic.txt").write_text(_render_text(payload))


def _render_text(payload: dict[str, Any]) -> str:
    lines = [
        f"N01 v1.0 analytic suite  ({payload['generated_at']})",
        f"python {payload['environment']['python']}  numpy {payload['environment']['numpy']}  "
        f"torch {payload['environment']['torch']}",
        "",
    ]
    for case in payload["cases"]:
        lines.append(f"[{case['status'].upper()}] {case['name']}")
        if case["notes"]:
            lines.append(f"    {case['notes']}")
    summary = payload["summary"]
    lines.append("")
    lines.append(f"{summary['passed']}/{summary['total']} cases passed")
    if summary["failed"]:
        lines.append(f"failed: {', '.join(summary['failed'])}")
    return "\n".join(lines) + "\n"


def analytic_command(args: argparse.Namespace) -> int:
    payload = run_analytic(args.case)
    output = Path(args.output)
    _write_receipt(payload, output)
    print(_render_text(payload), end="")
    print(f"receipt: {output / 'analytic.json'}")
    return 0 if not payload["summary"]["failed"] else 1


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="owac-n01-full-eval", description=__doc__)
    subcommands = top.add_subparsers(dest="command", required=True)
    analytic = subcommands.add_parser(
        "analytic", help="run the P1 numeric/behavioural analytic suite"
    )
    analytic.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/n01-p1-analytic"),
        help="receipt directory (default: outputs/n01-p1-analytic)",
    )
    analytic.add_argument(
        "--case",
        action="append",
        help="run only this case (repeatable); default runs every case",
    )
    analytic.set_defaults(handler=analytic_command)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

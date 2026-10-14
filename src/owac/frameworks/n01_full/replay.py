"""Replay: one full solve, two retraction interfaces, one dependency closure.

Implementation of ``docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md`` section 4 and
``P1_ANALYTIC_REFERENCE.md`` sections 1 and 3.  Three entry points exist because
"retraction succeeded" must not stay one ambiguous metric:

* :func:`full_replay` consumes an event stream once, commits the domains, the
  per-domain fits, the readout and a versioned
  :class:`~owac.frameworks.n01_full.contracts.ReadoutSnapshot`, and declares the
  exact replay range (at most ``ReplayConfiguration.exact_window_unique_frames``
  unique frames per camera; older exposures survive only as declared history
  loss -- unlimited exact retraction is never promised);
* :func:`fixed_decision_replay` freezes motion/support decisions and replaces the
  retracted source's *content*; a difference means the content path still reads
  the retracted source.  If a frozen decision was itself derived from that
  content, the caller must list it in ``frozen_decision_deps`` and the result is
  only conditional;
* :func:`counterfactual_replay` deletes the source from the stream, recomputes
  every decision, and may compare against an independent batch reference passed in
  as ``reference_fn`` (the oracle lives in ``reference.py`` and is never imported
  here, so the comparison cannot share a code path with production).

:class:`ToyProgram` is the two-layer nonlinear operator the stage prompt requires
for the dependency-closure property test: sources -> normalization -> centering ->
tanh hidden -> quadratic output -> readout.  Removing a source changes the
normalization, so "zero the removed input" and "recompute the successor closure"
provably differ, which is what makes the property test falsifiable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01_full.config import ReplayConfig
from owac.frameworks.n01_full.contracts import (
    DependencyNode,
    ReadoutSnapshot,
    ReplayRange,
    SourceKey,
)
from owac.frameworks.n01_full.domains import VersionedDomain, retract_sources
from owac.frameworks.n01_full.ledger import (
    ContentReplacement,
    FrameIdentity,
    MeasurementLedger,
)
from owac.frameworks.n01_full.statistics import StructureFit, fit_structure

__all__ = [
    "ObserveEvent",
    "ResetEvent",
    "LedgerEvent",
    "EventStream",
    "HistoryLoss",
    "QueryPoint",
    "ReadoutGrid",
    "StructureReadout",
    "FullReplayResult",
    "FixedDecisionReplayResult",
    "CounterfactualReplayResult",
    "consume",
    "structure_readout",
    "full_replay",
    "fixed_decision_replay",
    "counterfactual_replay",
    "ToyNode",
    "ToyProgram",
    "recompute_successor_closure",
]


@dataclass(frozen=True)
class ObserveEvent:
    """One camera's arrival event; the only event that can create a measurement."""

    frame: MeasurementFrame


@dataclass(frozen=True)
class ResetEvent:
    """A partial episode reset inside the stream (batch members are independent)."""

    episode_ids: tuple[str, ...]
    camera_id: str | None = None


LedgerEvent = ObserveEvent | ResetEvent


@dataclass(frozen=True)
class EventStream:
    """The input event stream of one replay, in causal order."""

    dataset_id: str
    events: tuple[LedgerEvent, ...]
    value_units: str = "unknown"
    calibration_version: str = "unknown"
    time_unit: str = "frame"

    def __len__(self) -> int:
        return len(self.events)


def consume(stream: EventStream) -> tuple[MeasurementLedger, tuple[dict[str, Any], ...]]:
    """Build the ledger from the stream, one arrival event at a time."""
    ledger = MeasurementLedger(
        stream.dataset_id,
        value_units=stream.value_units,
        calibration_version=stream.calibration_version,
        time_unit=stream.time_unit,
    )
    receipts: list[dict[str, Any]] = []
    for index, event in enumerate(stream.events):
        if isinstance(event, ObserveEvent):
            outcome = ledger.observe(event.frame)
            receipts.append(
                {
                    "event": index,
                    "kind": "observe",
                    "registered": [list(item) for item in outcome.registered_frames],
                    "duplicates": [list(item) for item in outcome.duplicate_frames],
                    "skipped": [list(item) for item in outcome.skipped_frames],
                    "new_samples": outcome.new_samples,
                }
            )
        elif isinstance(event, ResetEvent):
            result = ledger.reset(event.episode_ids, camera_id=event.camera_id)
            receipts.append({"event": index, "kind": "reset", **result})
        else:  # pragma: no cover - guarded by the type alias
            raise TypeError(f"unknown ledger event {type(event).__name__}")
    return ledger, tuple(receipts)


@dataclass(frozen=True)
class HistoryLoss:
    """One exposure that fell out of the exact replay window."""

    frame: FrameIdentity
    reason: str


class QueryPoint(NamedTuple):
    """A declared readout query: label, domain, phase and chart coordinate."""

    label: str
    domain_id: str
    phase: int | str
    x: float
    y: float


@dataclass(frozen=True)
class ReadoutGrid:
    """Query coordinates shared by every run that must be compared."""

    points: tuple[QueryPoint, ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(point.label for point in self.points)


@dataclass(frozen=True)
class StructureReadout:
    """The committed readout: per-query means and variances in declared units."""

    labels: tuple[str, ...]
    means: torch.Tensor
    variances: torch.Tensor
    domain_versions: tuple[str, ...]

    def max_abs_diff(self, other: StructureReadout) -> float:
        """Largest readout difference; mismatched validity/uncertainty is infinite."""
        if self.labels != other.labels:
            return float("inf")
        left = torch.cat([self.means, self.variances])
        right = torch.cat([other.means, other.variances])
        finite = torch.isfinite(left) & torch.isfinite(right)
        if not bool(torch.equal(torch.isfinite(left), torch.isfinite(right))):
            return float("inf")
        if not bool(finite.any()):
            return 0.0
        return float((left[finite] - right[finite]).abs().max())

    def as_vector(self) -> dict[str, float]:
        return {label: float(value) for label, value in zip(self.labels, self.means, strict=True)}


def structure_readout(fits: Sequence[StructureFit], grid: ReadoutGrid) -> StructureReadout:
    """Read every declared query from the fit of its domain (NaN when absent)."""
    by_domain: dict[str, StructureFit] = {}
    for fit in fits:
        by_domain[fit.domain_id] = fit
    means, variances = [], []
    for point in grid.points:
        fit = by_domain.get(point.domain_id)
        if fit is None:
            means.append(float("nan"))
            variances.append(float("nan"))
            continue
        mean, variance = fit.predict(point.phase, (point.x, point.y))
        means.append(mean)
        variances.append(variance)
    return StructureReadout(
        labels=grid.labels(),
        means=torch.tensor(means, dtype=torch.float64),
        variances=torch.tensor(variances, dtype=torch.float64),
        domain_versions=tuple(fit.support_version for fit in fits),
    )


@dataclass(frozen=True)
class FullReplayResult:
    """The committed outcome of one full solve."""

    ledger: MeasurementLedger
    domains: tuple[VersionedDomain, ...]
    fits: tuple[StructureFit, ...]
    readout: StructureReadout
    snapshot: ReadoutSnapshot
    exact_replay_range: ReplayRange
    exact_window_frames: tuple[FrameIdentity, ...]
    history_loss: tuple[HistoryLoss, ...]
    receipts: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _exact_window(
    ledger: MeasurementLedger, exact_window: int
) -> tuple[tuple[FrameIdentity, ...], tuple[HistoryLoss, ...]]:
    if exact_window < 1:
        raise ValueError("exact_window must keep at least one unique frame")
    selected: list[FrameIdentity] = []
    dropped: list[HistoryLoss] = []
    cameras = sorted(
        {(item.identity.episode_id, item.identity.camera_id) for item in ledger.frames()}
    )
    for episode_id, camera_id in cameras:
        frames = ledger.unique_frames(episode_id=episode_id, camera_id=camera_id)
        keep = frames[-exact_window:]
        selected.extend(keep)
        for identity in frames[: len(frames) - len(keep)]:
            dropped.append(HistoryLoss(frame=identity, reason="beyond_exact_window_summary_only"))
    selected.sort(key=lambda item: (item.episode_id, item.camera_id, _frame_order(item.frame_id)))
    dropped.sort(key=lambda item: (item.frame.episode_id, item.frame.camera_id))
    return tuple(selected), tuple(dropped)


def _frame_order(frame_id: int | str) -> tuple[int, int, str]:
    if isinstance(frame_id, int) and not isinstance(frame_id, bool):
        return (0, frame_id, "")
    return (1, 0, str(frame_id))


def _calibration_version(ledger: MeasurementLedger) -> str:
    versions = {frame.calibration_version for frame in ledger.frames()}
    if len(versions) == 1:
        return next(iter(versions))
    return "mixed" if versions else "empty"


def _support_version(domains: Sequence[VersionedDomain]) -> str:
    payload = [
        domain.support_version for domain in sorted(domains, key=lambda item: item.domain_id)
    ]
    return "svset:" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def full_replay(
    events: EventStream | Sequence[LedgerEvent],
    *,
    domain_builder: Callable[[MeasurementLedger], Sequence[VersionedDomain]] | None = None,
    domains: Sequence[VersionedDomain] | None = None,
    noise_model: Any | None = None,
    correspondence: Any | None = None,
    readout_grid: ReadoutGrid | None = None,
    replay_config: ReplayConfig | None = None,
    parameter_version: str | int = "analytic-p1",
    mapping_version: str | int = "identity",
    commit_time: str | int = "p1-analytic",
) -> FullReplayResult:
    """Solve the stream once and commit the result with its replay range."""
    stream = events if isinstance(events, EventStream) else EventStream("ledger", tuple(events))
    replay = replay_config or ReplayConfig()
    ledger, receipts = consume(stream)
    exact_frames, history_loss = _exact_window(ledger, replay.exact_window_unique_frames)
    ledger = ledger.retain_frames(exact_frames)
    if domain_builder is not None and domains is not None:
        raise ValueError("pass either a domain_builder or explicit domains, not both")
    if domain_builder is not None:
        built = tuple(domain_builder(ledger))
    elif domains is not None:
        built = tuple(domains)
    else:
        built = ()
    # Explicit domains also lose expired support; the resulting version must change.
    built = tuple(
        retract_sources(domain, [key for key in domain.members() if not ledger.has_source(key)])
        if any(not ledger.has_source(key) for key in domain.members())
        else domain
        for domain in built
    )
    fits = tuple(fit_structure(domain, ledger, correspondence, noise_model) for domain in built)
    grid = readout_grid or ReadoutGrid(())
    readout = structure_readout(fits, grid)
    frame_ids = [identity.frame_id for identity in exact_frames]
    exact_range = ReplayRange(
        oldest_frame_id=frame_ids[0] if frame_ids else "none",
        newest_frame_id=frame_ids[-1] if frame_ids else "none",
        unique_frames=max(len(exact_frames), 1),
    )
    snapshot = ReadoutSnapshot(
        representation=readout.means,
        validity=torch.isfinite(readout.means),
        source_time={
            "newest_frame": exact_range.newest_frame_id,
            "oldest_exact_frame": exact_range.oldest_frame_id,
            "unique_exact_frames": len(exact_frames),
        },
        commit_time=commit_time,
        versions={
            "parameter_version": parameter_version,
            "support_version": _support_version(built),
            "mapping_version": mapping_version,
            "calibration_version": _calibration_version(ledger),
        },
        exact_replay_range=exact_range,
        cost={
            "events": len(stream),
            "frames": ledger.frame_count(),
            "exact_window_frames": len(exact_frames),
            "history_loss_frames": len(history_loss),
            "domains": len(built),
            "fits": len(fits),
            "readout_queries": len(grid.points),
        },
    )
    manifest = {
        "kind": "full-replay",
        "dataset_id": stream.dataset_id,
        "events": len(stream),
        "registered_frames": ledger.frame_count(),
        "unique_sources": len({key for fit in fits for key in fit.estimate.evidence_source_ids}),
        "domains": [
            {
                "domain_id": domain.domain_id,
                "support_version": domain.support_version,
                "content_digest": domain.content_digest,
                "operation": domain.operation,
            }
            for domain in built
        ],
        "exact_replay_range": {
            "oldest": exact_range.oldest_frame_id,
            "newest": exact_range.newest_frame_id,
            "unique_frames": exact_range.unique_frames,
        },
        "history_loss": [tuple(loss.frame) for loss in history_loss],
        "beyond_exact_policy": replay.beyond_exact_policy,
        "versions": dict(snapshot.versions),
        "readout": readout.as_vector(),
    }
    return FullReplayResult(
        ledger=ledger,
        domains=built,
        fits=fits,
        readout=readout,
        snapshot=snapshot,
        exact_replay_range=exact_range,
        exact_window_frames=exact_frames,
        history_loss=history_loss,
        receipts=receipts,
        manifest=manifest,
    )


@dataclass(frozen=True)
class FixedDecisionReplayResult:
    """Outcome of the content-replacement replay under frozen decisions."""

    manifest: Mapping[str, Any]
    baseline_readout: StructureReadout
    replaced_readout: StructureReadout
    max_abs_diff: float
    unchanged: bool
    tolerance: float
    frozen_domain_versions: tuple[str, ...]
    conditional_on: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def fixed_decision_replay(
    events: EventStream | Sequence[LedgerEvent],
    *,
    frozen_domains: Sequence[VersionedDomain],
    retracted_ids: Sequence[SourceKey],
    replacement: ContentReplacement,
    noise_model: Any | None = None,
    correspondence: Any | None = None,
    readout_grid: ReadoutGrid | None = None,
    frozen_decision_deps: Sequence[str] = (),
    tolerance: float = 1e-10,
) -> FixedDecisionReplayResult:
    """Replace the retracted content under frozen decisions and compare readouts.

    The decisions (domain membership, basis, chart, noise version, mapping) are
    the post-retraction ones supplied by the caller.  A difference beyond
    ``tolerance`` means the content path still reads the retracted source, so the
    interface is falsifiable instead of self-confirming.
    """
    stream = events if isinstance(events, EventStream) else EventStream("ledger", tuple(events))
    ledger, _ = consume(stream)
    grid = readout_grid or ReadoutGrid(())
    if not grid.points:
        raise ValueError("fixed replay needs a nonempty query grid")
    if not retracted_ids or set(replacement.values) != set(retracted_ids):
        raise ValueError("replacement must cover exactly the nonempty retracted source set")
    if any(not ledger.has_source(key) for key in retracted_ids):
        raise ValueError("replacement targets must have been observed")
    if any(replacement.values[key] == ledger.get_source(key).raw_value for key in retracted_ids):
        raise ValueError("replacement must actually change every target's content")
    baseline = structure_readout(
        [fit_structure(domain, ledger, correspondence, noise_model) for domain in frozen_domains],
        grid,
    )
    # The replacement is a derived, versioned content view of the ledger; the raw
    # records stay untouched (see ContentReplacement / MeasurementLedger.with_content).
    replaced_ledger = ledger.with_content(replacement)
    replaced = structure_readout(
        [
            fit_structure(domain, replaced_ledger, correspondence, noise_model)
            for domain in frozen_domains
        ],
        grid,
    )
    diff = baseline.max_abs_diff(replaced)
    manifest = {
        "kind": "fixed-decision-replay",
        "replacement_id": replacement.replacement_id,
        "replacement_version": replacement.version,
        "retracted_ids": [list(key.identity()) for key in retracted_ids],
        "replaced_sources": len(replacement.values),
        "frozen_domain_versions": [domain.support_version for domain in frozen_domains],
        "compared": "readout means, variances and validity at the declared query grid",
        "tolerance": tolerance,
        "max_abs_diff": diff,
        "unchanged": diff <= tolerance,
        "conditional_on": list(frozen_decision_deps),
        "conclusion": (
            "content path no longer reads the retracted source under the frozen decisions"
            if diff <= tolerance
            else "content path still depends on the retracted source"
        ),
        "not_claimed": ["whole-result independence (that needs counterfactual_replay)"],
    }
    return FixedDecisionReplayResult(
        manifest=manifest,
        baseline_readout=baseline,
        replaced_readout=replaced,
        max_abs_diff=diff,
        unchanged=diff <= tolerance,
        tolerance=tolerance,
        frozen_domain_versions=tuple(domain.support_version for domain in frozen_domains),
        conditional_on=tuple(frozen_decision_deps),
    )


@dataclass(frozen=True)
class CounterfactualReplayResult:
    """Outcome of deleting the source from the stream and recomputing everything."""

    manifest: Mapping[str, Any]
    result: FullReplayResult
    retracted_ids: tuple[SourceKey, ...]
    beyond_exact_range: bool
    history_loss: tuple[HistoryLoss, ...]
    claim: str
    reference_max_abs_diff: float | None
    matches_reference: bool | None
    reference_tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def counterfactual_replay(
    events: EventStream | Sequence[LedgerEvent],
    *,
    domain_builder: Callable[[MeasurementLedger], Sequence[VersionedDomain]],
    retracted_ids: Sequence[SourceKey],
    noise_model: Any | None = None,
    correspondence: Any | None = None,
    readout_grid: ReadoutGrid | None = None,
    replay_config: ReplayConfig | None = None,
    reference_fn: Callable[
        [MeasurementLedger, Sequence[VersionedDomain], ReadoutGrid], Mapping[str, float]
    ]
    | None = None,
    reference_tolerance: tuple[float, float] = (1e-10, 1e-8),
    parameter_version: str | int = "analytic-p1",
    mapping_version: str | int = "identity",
    commit_time: str | int = "p1-analytic",
    correspondence_builder: Callable[[MeasurementLedger], Any] | None = None,
    noise_model_builder: Callable[[MeasurementLedger], Any] | None = None,
) -> CounterfactualReplayResult:
    """Delete the sources, recompute every decision, and compare to the reference.

    The claim level is deliberately limited: a source outside the exact replay
    window is rebuilt from the legal boundary with the contaminated summary
    invalidated, and no exact equivalence is claimed for it.
    """
    retracted = tuple(retracted_ids)
    stream = events if isinstance(events, EventStream) else EventStream("ledger", tuple(events))
    replay = replay_config or ReplayConfig()
    full, _ = consume(stream)
    window_frames, window_loss = _exact_window(full, replay.exact_window_unique_frames)
    window_keys = set(window_frames)
    untouched = [
        key
        for key in retracted
        if FrameIdentity(full.dataset_id, key.episode_id, key.camera_id, key.frame_id)
        not in window_keys
    ]
    if any(not full.has_source(key) for key in retracted):
        raise ValueError("counterfactual retraction targets must have been observed")
    view = full.retain_frames(window_frames).without(retracted)
    if correspondence_builder is not None:
        if correspondence is not None:
            raise ValueError("pass correspondence or its recomputation builder, not both")
        correspondence = correspondence_builder(view)
    if noise_model_builder is not None:
        if noise_model is not None:
            raise ValueError("pass noise_model or its recomputation builder, not both")
        noise_model = noise_model_builder(view)
    domains = tuple(domain_builder(view))
    fits = tuple(fit_structure(domain, view, correspondence, noise_model) for domain in domains)
    grid = readout_grid or ReadoutGrid(())
    readout = structure_readout(fits, grid)
    snapshot = ReadoutSnapshot(
        representation=readout.means,
        validity=torch.isfinite(readout.means),
        source_time={
            "oldest_exact_frame": window_frames[0].frame_id if window_frames else "none",
            "newest_exact_frame": window_frames[-1].frame_id if window_frames else "none",
        },
        commit_time=commit_time,
        versions={
            "parameter_version": parameter_version,
            "support_version": _support_version(domains),
            "mapping_version": mapping_version,
            "calibration_version": _calibration_version(view),
        },
        exact_replay_range=ReplayRange(
            oldest_frame_id=window_frames[0].frame_id if window_frames else "none",
            newest_frame_id=window_frames[-1].frame_id if window_frames else "none",
            unique_frames=max(len(window_frames), 1),
        ),
        cost={
            "events": len(stream),
            "retracted_sources": len(retracted),
            "domains": len(domains),
            "fits": len(fits),
            "readout_queries": len(grid.points),
        },
    )
    rebuilt = FullReplayResult(
        ledger=view,
        domains=domains,
        fits=fits,
        readout=readout,
        snapshot=snapshot,
        exact_replay_range=snapshot.exact_replay_range,
        exact_window_frames=window_frames,
        history_loss=window_loss,
        receipts=(),
        manifest={
            "kind": "counterfactual-replay",
            "retracted_sources": [list(key.identity()) for key in retracted],
            "domains": [domain.support_version for domain in domains],
            "readout": readout.as_vector(),
        },
    )

    reference_diff: float | None = None
    matches: bool | None = None
    reference_detail: dict[str, Any] = {}
    if reference_fn is not None:
        reference = dict(reference_fn(view, domains, grid))
        if not reference:
            raise ValueError("the independent reference returned no comparable values")
        atol, rtol = reference_tolerance
        observed_vector = readout.as_vector()
        if set(reference) != set(observed_vector):
            raise ValueError("reference must cover the entire query grid")
        worst = 0.0
        inside = True
        for label, value in reference.items():
            observed = observed_vector.get(label)
            if observed is None:
                inside = False
                worst = float("inf")
                break
            difference = abs(observed - float(value))
            worst = max(worst, difference)
            inside = inside and difference <= atol + rtol * abs(float(value))
        reference_diff = worst
        matches = inside
        reference_detail = {
            "reference_labels": sorted(reference),
            "tolerance": {"atol": atol, "rtol": rtol},
        }

    beyond = bool(untouched)
    claim = "rebuild_from_legal_boundary" if beyond else "exact_independence_in_event_stream"
    manifest = {
        "kind": "counterfactual-replay",
        "retracted_sources": [list(key.identity()) for key in retracted],
        "recomputed": [
            "ledger view",
            "domain membership and versions",
            "per-phase designs and noise sub-blocks",
            "readout",
        ],
        "beyond_exact_range": beyond,
        "beyond_exact_sources": [list(key.identity()) for key in untouched],
        "history_loss": [tuple(item.frame) for item in window_loss],
        "invalidated_summary": bool(untouched),
        "history_policy": "discard expired raw and summaries; rebuild from retained boundary",
        "decision_scope": "analytic callbacks recomputed on redacted bounded ledger",
        "correspondence_recomputed": correspondence_builder is not None,
        "noise_recomputed": noise_model_builder is not None,
        "claim": claim,
        "not_claimed": []
        if not beyond
        else ["exact equivalence for sources outside the exact replay window"],
        "reference_max_abs_diff": reference_diff,
        "matches_reference": matches,
        **reference_detail,
    }
    return CounterfactualReplayResult(
        manifest=manifest,
        result=rebuilt,
        retracted_ids=retracted,
        beyond_exact_range=beyond,
        history_loss=window_loss,
        claim=claim,
        reference_max_abs_diff=reference_diff,
        matches_reference=matches,
        reference_tolerance=reference_tolerance[0],
    )


# ------------------------------------------------------------------------------
# Toy two-layer nonlinear operator for the dependency-closure property test
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyNode:
    """One node of the toy dependency graph: parents and a compute rule."""

    node_id: str
    kind: str
    parent_ids: tuple[str, ...]


class ToyProgram:
    """Sources -> normalization -> centering -> tanh hidden -> quadratic output.

    Used only to exercise dependency-closure retraction (T5): the recomputation
    must follow ``parent_ids`` and the normalization makes the difference between
    "recompute the closure" and "zero the removed input" observable.
    """

    def __init__(
        self,
        source_ids: Sequence[str],
        *,
        hidden: torch.Tensor,
        output: torch.Tensor,
        readout_weights: torch.Tensor | None = None,
        parameter_version: str = "toy-p1",
        calibration_version: str = "toy-cal",
        mapping_version: str = "toy-map",
        support_version: str = "toy-support",
    ) -> None:
        if not source_ids:
            raise ValueError("the toy operator needs at least one source")
        self.source_ids = tuple(source_ids)
        self.hidden = hidden.to(torch.float64)
        self.output = output.to(torch.float64)
        if self.hidden.ndim != 2 or self.hidden.shape[1] != len(self.source_ids):
            raise ValueError("toy hidden weights must be [H, n_sources]")
        if self.output.ndim != 2 or self.output.shape[1] != self.hidden.shape[0]:
            raise ValueError("toy output weights must be [O, H]")
        self.readout_weights = (
            torch.ones(self.output.shape[0], dtype=torch.float64)
            if readout_weights is None
            else readout_weights.to(torch.float64)
        )
        self.versions = {
            "parameter_version": parameter_version,
            "support_version": support_version,
            "mapping_version": mapping_version,
            "calibration_version": calibration_version,
        }

    # -- graph --------------------------------------------------------------
    def nodes(self) -> dict[str, ToyNode]:
        nodes: dict[str, ToyNode] = {
            source: ToyNode(source, "toy.source", ()) for source in self.source_ids
        }
        nodes["normalization"] = ToyNode("normalization", "toy.normalization", self.source_ids)
        for index, source in enumerate(self.source_ids):
            nodes[f"centered.{index}"] = ToyNode(
                f"centered.{index}", "toy.centered", (source, "normalization")
            )
        nodes["hidden"] = ToyNode(
            "hidden",
            "toy.hidden",
            tuple(f"centered.{index}" for index in range(len(self.source_ids))),
        )
        nodes["output"] = ToyNode("output", "toy.output", ("hidden",))
        nodes["readout"] = ToyNode("readout", "toy.readout", ("output",))
        return nodes

    def dependency_nodes(self) -> dict[str, DependencyNode]:
        """The same graph expressed with the contract's :class:`DependencyNode`."""
        return {
            node.node_id: DependencyNode(
                node_id=node.node_id,
                kind=node.kind,
                parent_ids=node.parent_ids,
                **self.versions,
            )
            for node in self.nodes().values()
        }

    def _order(self) -> tuple[str, ...]:
        order = list(self.source_ids)
        order.append("normalization")
        order.extend(f"centered.{index}" for index in range(len(self.source_ids)))
        order.extend(["hidden", "output", "readout"])
        return tuple(order)

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, values: Mapping[str, float]) -> dict[str, torch.Tensor]:
        """Evaluate the whole program from the given source values."""
        missing = [source for source in self.source_ids if source not in values]
        if missing:
            raise KeyError(f"missing source values for {missing}")
        present = torch.tensor(
            [float(values[source]) for source in self.source_ids], dtype=torch.float64
        )
        return self._forward(present)

    def _forward(self, present: torch.Tensor) -> dict[str, torch.Tensor]:
        normalization = present.mean()
        centered = present - normalization
        hidden = torch.tanh(self.hidden @ centered)
        output = self.output @ (hidden * hidden)
        readout = self.readout_weights @ output
        result = {source: present[index] for index, source in enumerate(self.source_ids)}
        result["normalization"] = normalization
        for index in range(len(self.source_ids)):
            result[f"centered.{index}"] = centered[index]
        result["hidden"] = hidden
        result["output"] = output
        result["readout"] = readout
        return result

    # -- retraction ---------------------------------------------------------
    def retract(
        self, values: Mapping[str, float], retracted: Iterable[str]
    ) -> dict[str, torch.Tensor]:
        """Full recomputation over the surviving sources (the batch reference)."""
        drop = set(retracted)
        unknown = drop - set(self.source_ids)
        if unknown:
            raise KeyError(f"cannot retract unknown toy sources {sorted(unknown)}")
        survivors = [source for source in self.source_ids if source not in drop]
        if not survivors:
            raise ValueError("retracting every source leaves no measurement")
        present = torch.tensor([float(values[source]) for source in survivors], dtype=torch.float64)
        reduced = ToyProgram(
            survivors,
            hidden=self.hidden[:, [self.source_ids.index(source) for source in survivors]],
            output=self.output,
            readout_weights=self.readout_weights,
            **self.versions,
        )
        return reduced._forward(present)

    def zero_input(
        self, values: Mapping[str, float], retracted: Iterable[str]
    ) -> dict[str, torch.Tensor]:
        """Documented wrong version: keep the graph, zero the removed input.

        It changes the normalization exactly like deleting the *value* would, so
        it is not the same as removing the source from the structure; the
        property test asserts it differs or is caught.
        """
        zeroed = dict(values)
        for source in retracted:
            if source not in zeroed:
                raise KeyError(f"cannot zero unknown toy source {source!r}")
            zeroed[source] = 0.0
        return self.evaluate(zeroed)


def recompute_successor_closure(
    program: ToyProgram,
    previous: Mapping[str, Any],
    retracted: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Recompute exactly the invalidated successor closure, topologically.

    Returns ``(values, recomputed_node_ids)``.  Nodes outside the closure keep
    their previous values; a node inside it is recomputed from its parents, so
    the property test can compare against a full recomputation.
    """
    nodes = program.nodes()
    for source in retracted:
        if source not in nodes:
            raise KeyError(f"cannot retract unknown toy node {source!r}")
    invalidated: set[str] = set(retracted)
    changed = True
    while changed:
        changed = False
        for node in nodes.values():
            if node.node_id in invalidated:
                continue
            if any(parent in invalidated for parent in node.parent_ids):
                invalidated.add(node.node_id)
                changed = True
    values = dict(previous)
    recomputed: list[str] = []
    for node_id in program._order():
        if node_id not in invalidated:
            continue
        node = nodes[node_id]
        if node.kind == "toy.source":
            values.pop(node_id, None)
            recomputed.append(node_id)
            continue
        surviving = [
            values[parent]
            for parent in node.parent_ids
            if parent in values and parent not in invalidated
        ]
        if node.kind == "toy.normalization":
            values[node_id] = torch.stack(surviving).mean()
        elif node.kind == "toy.centered":
            source, normalization = node.parent_ids
            if source not in values:
                values.pop(node_id, None)
                recomputed.append(node_id)
                continue
            values[node_id] = values[source] - values[normalization]
        elif node.kind == "toy.hidden":
            survivors = [
                (index, parent)
                for index, parent in enumerate(node.parent_ids)
                if values.get(parent) is not None
            ]
            columns = [index for index, _ in survivors]
            centered = torch.stack([values[parent] for _, parent in survivors])
            values[node_id] = torch.tanh(program.hidden[:, columns] @ centered)
        elif node.kind == "toy.output":
            hidden = values["hidden"]
            values[node_id] = program.output @ (hidden * hidden)
        elif node.kind == "toy.readout":
            values[node_id] = program.readout_weights @ values["output"]
        else:  # pragma: no cover - guarded by nodes()
            raise ValueError(f"unknown toy node kind {node.kind!r}")
        recomputed.append(node_id)
    return values, tuple(recomputed)

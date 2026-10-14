"""Support domains: make / merge / split / retract, each returning a new version.

Design of ``docs/prompts/n01_v1_0/P1_ANALYTIC_REFERENCE.md`` sections 1 and 3:
every operation is a pure function of already-committed objects, so a retraction
never mutates the parent domain, never touches the ledger and never reaches a
different domain that still legally references the source.

Two identifiers are carried per version on purpose:

* ``support_version`` names the lineage (operation and parent versions) plus the
  committed content, so *every* change produces a new version and derived nodes
  can record their full parents;
* ``content_digest`` names only the committed membership (canonical evidence and
  context sets, basis, chart, phase, segment, decision/mapping version).  A
  retract-then-restore therefore has a different lineage but the same content,
  which is what the set-algebra checks compare.

Merging is deliberately not a compatibility judge: it refuses domains whose
declared basis/chart/phase/segment/decision/mapping versions differ instead of
guessing, because P1 fixes legal domains from the test side and autonomous
compatibility decisions belong to P3.  Roles are preserved: a source that is
evidence in one parent and context in another is a role conflict and is refused
rather than silently reclassified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from owac.frameworks.n01_full.contracts import SUPPORT_ROLES, SourceKey, SupportDomain

__all__ = [
    "VersionedDomain",
    "canonical_source_key",
    "canonical_sources",
    "make_domain",
    "merge_domains",
    "split_domain",
    "retract_sources",
    "restore_sources",
]

#: Declaration fields merge refuses to average over.  Two domains may be merged
#: only when every one of these agrees; P1 fixes legal domains from the test side.
MERGE_FIXED_FIELDS = (
    "reference_chart",
    "basis",
    "phase",
    "temporal_segment_id",
    "decision_version",
    "mapping_version",
)


def canonical_source_key(source: SourceKey) -> tuple[Any, ...]:
    """Total, deterministic order over identities (declared lexicographic order).

    Numbers sort numerically before strings so ``frame_id=2`` precedes
    ``frame_id=10`` instead of sorting as text; mixed int/str identities still
    have a total order because the type tag precedes the value.
    """

    def frame_part(value: int | str) -> tuple[int, float, str]:
        if isinstance(value, int) and not isinstance(value, bool):
            return (0, float(value), "")
        return (1, 0.0, str(value))

    def phase_part(value: int | str) -> tuple[int, float, str]:
        if isinstance(value, int) and not isinstance(value, bool):
            return (0, float(value), "")
        return (1, 0.0, str(value))

    return (
        source.dataset_id,
        source.episode_id,
        source.camera_id,
        frame_part(source.frame_id),
        phase_part(source.channel_or_phase),
        source.sensor_y,
        source.sensor_x,
    )


def canonical_sources(
    sources: Iterable[SourceKey],
) -> tuple[tuple[SourceKey, ...], tuple[SourceKey, ...]]:
    """Return ``(canonical unique sources, duplicates in input order)``.

    One RAW sample is registered once per estimate: a repeated identity does not
    become a second independent observation (contract section 3).
    """
    unique: dict[tuple[Any, ...], SourceKey] = {}
    duplicates: list[SourceKey] = []
    for source in sources:
        if not isinstance(source, SourceKey):
            raise TypeError("domain membership must contain SourceKey instances")
        identity = source.identity()
        if identity in unique:
            duplicates.append(source)
            continue
        unique[identity] = source
    ordered = tuple(sorted(unique.values(), key=canonical_source_key))
    return ordered, tuple(duplicates)


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _domain_payload(domain: SupportDomain) -> dict[str, Any]:
    return {
        "domain_id": domain.domain_id,
        "reference_chart": domain.reference_chart,
        "basis": domain.basis,
        "phase": None if domain.phase is None else str(domain.phase),
        "temporal_segment_id": str(domain.temporal_segment_id),
        "decision_version": domain.decision_version,
        "mapping_version": domain.mapping_version,
        "evidence_ids": [list(source.identity()) for source in domain.evidence_source_ids],
        "context_ids": [list(source.identity()) for source in domain.context_source_ids],
    }


@dataclass(frozen=True)
class VersionedDomain:
    """A committed support domain plus the versions it was committed under."""

    domain: SupportDomain
    support_version: str
    content_digest: str
    operation: str
    parent_versions: tuple[str, ...] = ()
    bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.support_version:
            raise ValueError("VersionedDomain.support_version must be declared")
        if not self.content_digest:
            raise ValueError("VersionedDomain.content_digest must be declared")
        if not self.operation:
            raise ValueError("VersionedDomain.operation must name the producing operation")
        if not isinstance(self.parent_versions, tuple):
            raise TypeError("VersionedDomain.parent_versions must be a tuple")
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    # -- convenience --------------------------------------------------------
    @property
    def domain_id(self) -> str:
        return self.domain.domain_id

    @property
    def evidence_source_ids(self) -> tuple[SourceKey, ...]:
        return self.domain.evidence_source_ids

    @property
    def context_source_ids(self) -> tuple[SourceKey, ...]:
        return self.domain.context_source_ids

    def members(self) -> tuple[SourceKey, ...]:
        """Every source the domain references, canonical order, roles merged out."""
        ordered, _ = canonical_sources(
            [*self.domain.evidence_source_ids, *self.domain.context_source_ids]
        )
        return ordered

    def role_of(self, source: SourceKey) -> str | None:
        return self.domain.role_of(source)

    def evidence_count(self) -> int:
        return len(self.domain.evidence_source_ids)

    def binding(self, name: str) -> str | None:
        return self.bindings.get(name)


def _commit(
    domain: SupportDomain,
    *,
    operation: str,
    parent_versions: Sequence[str],
    bindings: Mapping[str, str],
) -> VersionedDomain:
    domain = replace(
        domain,
        evidence_source_ids=canonical_sources(domain.evidence_source_ids)[0],
        context_source_ids=canonical_sources(domain.context_source_ids)[0],
    )
    content = _digest({"domain": _domain_payload(domain)})
    parents = tuple(sorted(parent_versions))
    version = _digest(
        {
            "operation": operation,
            "parents": list(parents),
            "content": content,
            "bindings": dict(sorted(bindings.items())),
        }
    )
    return VersionedDomain(
        domain=domain,
        support_version=f"sv1-{version[:16]}",
        content_digest=f"cd1-{content[:16]}",
        operation=operation,
        parent_versions=parents,
        bindings=bindings,
    )


def make_domain(
    domain_id: str,
    *,
    reference_chart: str,
    basis: str,
    phase: int | str | None,
    evidence_source_ids: Iterable[SourceKey],
    context_source_ids: Iterable[SourceKey] = (),
    temporal_segment_id: str | int = "segment-0",
    decision_version: str,
    mapping_version: str,
    noise_version: str | None = None,
    prior_id: str | None = None,
) -> VersionedDomain:
    """Create a new domain version from explicitly declared members."""
    bindings = _bindings(noise_version=noise_version, prior_id=prior_id)
    domain = SupportDomain(
        domain_id=domain_id,
        reference_chart=reference_chart,
        basis=basis,
        phase=phase,
        evidence_source_ids=tuple(evidence_source_ids),
        context_source_ids=tuple(context_source_ids),
        temporal_segment_id=temporal_segment_id,
        decision_version=decision_version,
        mapping_version=mapping_version,
    )
    return _commit(domain, operation="make", parent_versions=(), bindings=bindings)


def merge_domains(
    domains: Iterable[VersionedDomain],
    *,
    domain_id: str | None = None,
    noise_version: str | None = None,
    prior_id: str | None = None,
) -> VersionedDomain:
    """Union the sources of fixed-version domains, canonically ordered, rebuilt."""
    parents = tuple(domains)
    if len(parents) < 2:
        raise ValueError("merge_domains needs at least two parent domains")
    for parent in parents:
        if not isinstance(parent, VersionedDomain):
            raise TypeError("merge_domains parents must be VersionedDomain instances")
    head = parents[0].domain
    for parent in parents[1:]:
        for name in MERGE_FIXED_FIELDS:
            left = getattr(head, name)
            right = getattr(parent.domain, name)
            if left != right:
                raise ValueError(
                    f"merge refused: {name} differs between {head.domain_id!r} ({left!r}) and "
                    f"{parent.domain_id!r} ({right!r}); P1 merges only fixed-version domains"
                )
    declared = _bindings(noise_version=noise_version, prior_id=prior_id)
    for parent in parents:
        for name, value in parent.bindings.items():
            if name in declared and declared[name] != value:
                raise ValueError(
                    f"merge refused: {name} {declared[name]!r} contradicts parent "
                    f"{parent.support_version} value {value!r}"
                )
            declared.setdefault(name, value)

    roles: dict[tuple[Any, ...], tuple[SourceKey, str]] = {}
    for parent in parents:
        for source in parent.domain.evidence_source_ids:
            _register_role(roles, source, "evidence")
        for source in parent.domain.context_source_ids:
            _register_role(roles, source, "context")

    evidence = tuple(key for key, role in roles.values() if role == "evidence")  # type: ignore[misc]
    context = tuple(key for key, role in roles.values() if role == "context")  # type: ignore[misc]
    merged_id = domain_id or "+".join(sorted({parent.domain_id for parent in parents}))
    domain = SupportDomain(
        domain_id=merged_id,
        reference_chart=head.reference_chart,
        basis=head.basis,
        phase=head.phase,
        evidence_source_ids=canonical_sources(evidence)[0],
        context_source_ids=canonical_sources(context)[0],
        temporal_segment_id=head.temporal_segment_id,
        decision_version=head.decision_version,
        mapping_version=head.mapping_version,
    )
    return _commit(
        domain,
        operation="merge",
        parent_versions=[parent.support_version for parent in parents],
        bindings=declared,
    )


def split_domain(
    parent: VersionedDomain,
    groups: Sequence[tuple[str, Sequence[SourceKey]]],
    *,
    decision_version: str | None = None,
) -> tuple[VersionedDomain, ...]:
    """Partition one domain into child domains, each refit from its own members.

    The parent's estimate is never copied: a child version is committed with the
    child's own membership, so a later fit is computed from those sources only.
    """
    if not isinstance(parent, VersionedDomain):
        raise TypeError("split_domain expects a VersionedDomain")
    if not groups:
        raise ValueError("split_domain needs at least one child group")
    membership = set(parent.members())
    covered: set[SourceKey] = set()
    children: list[VersionedDomain] = []
    for child_id, members in groups:
        selected = tuple(members)
        if not selected:
            raise ValueError(f"split child {child_id!r} is empty")
        unique, _ = canonical_sources(selected)
        if len(unique) != len(tuple(selected)):
            raise ValueError(f"split child {child_id!r} lists a duplicate source")
        outside = [source for source in unique if source not in membership]
        if outside:
            raise ValueError(
                f"split child {child_id!r} lists {len(outside)} source(s) that are not in "
                f"parent {parent.domain_id!r}"
            )
        overlap = covered & set(unique)
        if overlap:
            raise ValueError(f"split children overlap on {len(overlap)} source(s)")
        covered |= set(unique)
        domain = replace(
            parent.domain,
            domain_id=child_id,
            decision_version=(
                parent.domain.decision_version if decision_version is None else decision_version
            ),
            evidence_source_ids=tuple(
                source for source in unique if parent.role_of(source) == "evidence"
            ),
            context_source_ids=tuple(
                source for source in unique if parent.role_of(source) == "context"
            ),
        )
        children.append(
            _commit(
                domain,
                operation="split",
                parent_versions=[parent.support_version],
                bindings=parent.bindings,
            )
        )
    if covered != membership:
        missing = sorted(membership - covered, key=canonical_source_key)
        raise ValueError(
            f"split must partition the parent domain; {len(missing)} source(s) are unassigned "
            f"(first: {missing[0].identity()})"
        )
    return tuple(children)


def retract_sources(parent: VersionedDomain, source_ids: Iterable[SourceKey]) -> VersionedDomain:
    """Remove sources from the current domain version; raw records stay untouched."""
    if not isinstance(parent, VersionedDomain):
        raise TypeError("retract_sources expects a VersionedDomain")
    targets, duplicates = canonical_sources(source_ids)
    if duplicates:
        raise ValueError("retract_sources lists the same source twice")
    membership = set(parent.members())
    unknown = [source for source in targets if source not in membership]
    if unknown:
        raise ValueError(
            f"retract_sources lists {len(unknown)} source(s) that are not in domain "
            f"{parent.domain_id!r} (first: {unknown[0].identity()}); retraction is not a "
            "silent no-op"
        )
    remaining = [source for source in parent.members() if source not in set(targets)]
    roles = {source: parent.role_of(source) for source in remaining}
    domain = replace(
        parent.domain,
        evidence_source_ids=tuple(s for s in remaining if roles[s] == "evidence"),
        context_source_ids=tuple(s for s in remaining if roles[s] == "context"),
    )
    return _commit(
        domain,
        operation="retract",
        parent_versions=[parent.support_version],
        bindings=parent.bindings,
    )


def restore_sources(
    parent: VersionedDomain,
    source_ids: Iterable[SourceKey],
    *,
    role: str = "evidence",
) -> VersionedDomain:
    """Inverse of :func:`retract_sources`: add sources back with a declared role."""
    if not isinstance(parent, VersionedDomain):
        raise TypeError("restore_sources expects a VersionedDomain")
    if role not in SUPPORT_ROLES:
        raise ValueError(f"restore role must be one of {SUPPORT_ROLES}, got {role!r}")
    restored, duplicates = canonical_sources(source_ids)
    if duplicates:
        raise ValueError("restore_sources lists the same source twice")
    existing = set(parent.members())
    clash = [source for source in restored if source in existing]
    if clash:
        raise ValueError(
            f"restore_sources lists {len(clash)} source(s) already in domain "
            f"{parent.domain_id!r}; restoring an existing member is not a no-op change"
        )
    evidence = list(parent.domain.evidence_source_ids)
    context = list(parent.domain.context_source_ids)
    if role == "evidence":
        evidence.extend(restored)
    else:
        context.extend(restored)
    domain = replace(
        parent.domain,
        evidence_source_ids=tuple(evidence),
        context_source_ids=tuple(context),
    )
    return _commit(
        domain,
        operation="restore",
        parent_versions=[parent.support_version],
        bindings=parent.bindings,
    )


def _register_role(
    roles: dict[tuple[Any, ...], tuple[SourceKey, str]], source: SourceKey, role: str
) -> None:
    identity = source.identity()
    existing = roles.get(identity)
    if existing is not None and existing[1] != role:
        raise ValueError(
            f"role conflict on {identity}: {existing[1]} in one parent and {role} in another; "
            "roles are preserved and P1 does not silently reclassify a source"
        )
    roles[identity] = (source, role)


def _bindings(*, noise_version: str | None, prior_id: str | None) -> dict[str, str]:
    bindings: dict[str, str] = {}
    if noise_version is not None:
        if not noise_version:
            raise ValueError("noise_version must be a non-empty declaration")
        bindings["noise_version"] = noise_version
    if prior_id is not None:
        if not prior_id:
            raise ValueError("prior_id must be a non-empty declaration")
        bindings["prior_id"] = prior_id
    return bindings

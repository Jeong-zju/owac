"""Shared external contract for OWAC encoder candidates (v0 interoperability).

This module is deliberately lightweight: it must stay importable without torch,
numpy or any device initialization, because ``import owac`` must not create a
model or touch hardware. Fields carry concrete semantics; candidates keep their
internal state free (a state field, a hypothesis set, a graph, coefficients).

``docs/prompts/DeepSeek_N01_implementation.md`` fixes the field meaning.  The
shared external convention is::

    observations + metadata + optional_embodiment_context + optional_state + compute_budget
        -> representation + optional_state + validity + cost_record
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MeasurementFrame",
    "EmbodimentContext",
    "ComputeBudget",
    "EncoderOutput",
    "VisualTokens",
]


@dataclass
class MeasurementFrame:
    """One camera's calibrated measurement at one time step.

    ``values`` uses the raw numeric unit declared in ``metadata``; it is never an
    RGB/sRGB/XYZ intermediate image on the main path.  ``sample_xy`` holds the
    per-channel sampling coordinate/offset (for CFA this is the sub-grid phase);
    resolutions from different grids are never mixed in one frame.
    """

    values: Any
    valid_mask: Any | None = None
    bad_mask: Any | None = None
    saturated_mask: Any | None = None
    sample_xy: Any | None = None
    episode_ids: Any | None = None
    frame_ids: Any | None = None
    camera_id: str = ""
    timestamp: Any | None = None
    time_unit: str = "frame"
    new_measurement: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_available: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbodimentContext:
    """Optional body/task context.  Candidates declare which fields they use."""

    proprio: Any | None = None
    proprio_available: Any | None = None
    executed_action: Any | None = None
    executed_action_available: Any | None = None
    camera_transform: Any | None = None
    camera_transform_available: Any | None = None
    task_tokens: Any | None = None
    task_mask: Any | None = None


@dataclass
class ComputeBudget:
    """Deterministic update budget.

    ``deadline_ms`` is an optional *measured* soft budget; the first round
    controls cost through deterministic update counts and never calls Python
    clock polling a hard real-time guarantee.
    """

    max_updates: int = 1
    max_history: int = 8
    deadline_ms: float | None = None


@dataclass
class EncoderOutput:
    representation: Any
    state: Any
    validity: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualTokens:
    """Backend readout tokens: ``values`` is [B, L, D], ``mask`` is [B, L]."""

    values: Any
    mask: Any
    camera_ids: list[str] = field(default_factory=list)
    positions: Any | None = None

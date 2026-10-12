"""Shared v0 encoder contract for the N01/N02/N04 RAW encoder prototypes.

This module is the lightweight interoperation contract the three independent
encoder branches implement. It fixes field *semantics*, not internal state
layouts: a candidate may keep a state field, a hypothesis set, a particle cloud
or an analytic solver behind :meth:`Encoder.init_state` / :meth:`Encoder.step`.

Deliberate engineering choices:

* No third-party imports at module scope. ``import owac`` must not load torch,
  numpy or a device. Tensor-shaped fields are therefore typed as ``Any`` and the
  implementations attach their own tensors. The dataclasses validate only
  cheap structural facts so they stay usable in that light import path.
* The contract is intentionally small. Everything a candidate needs for its own
  computation lives in its own namespace; only what crosses the boundary
  (measurements in, representation and cost out) is fixed here.

Vocabulary used throughout:

``measurement``
    A raw sensor reading plus the sampling condition that gives it meaning
    (CFA phase, exposure, black/white level, timestamp). A measurement is a
    fact about the sensor, never a model output.
``observation``
    One frame's measurement set (one or more cameras) at one time step.
``prediction``
    A model's guess of what a measurement would be. It is compared with
    measurements and must never be fed back into the measurement buffer as if
    it had been sampled.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CameraSpec",
    "ComputeBudget",
    "CostRecord",
    "EmbodimentContext",
    "Encoder",
    "EncoderOutput",
    "MeasurementFrame",
    "Readout",
    "Validity",
    "VisualTokens",
]


def _as_tuple(value: Any, name: str) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if value is None:
        return ()
    raise TypeError(f"{name} must be a tuple or list, got {type(value).__name__}")


@dataclass(slots=True)
class MeasurementFrame:
    """One camera's measurement for one time step, in raw numeric units.

    ``values`` is ``[B, C, H, W]`` for a batched frame or ``[C, H, W]``/``[H, W]``
    for a single sample; the implementation module owns the tensor type. Units
    are declared in ``metadata`` (for example ``raw_units='uint8_pseudo_raw'``
    with ``white_level=255``) and are never assumed to be normalized [0, 1].

    ``sample_xy`` carries the per-channel sampling coordinate/phase grid. For
    ``bayer_cfa`` it encodes the CFA phase of each channel plane
    (RGGB/BGGR/GRBG/GBRG) and the sub-pixel offsets; for ``rawvla_rgb3`` it is
    empty or all-zero because the three channels are co-sited by construction.
    Planes of different spatial resolution must not be mixed into one tensor.
    """

    values: Any
    valid_mask: Any = None
    bad_mask: Any = None
    saturated_mask: Any = None
    sample_xy: Any = None
    episode_ids: tuple = ()
    frame_ids: tuple = ()
    camera_id: str = ""
    timestamp: Any = None
    time_unit: str = "frame"
    new_measurement: Any = None
    metadata: dict = field(default_factory=dict)
    metadata_available: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.episode_ids = _as_tuple(self.episode_ids, "episode_ids")
        self.frame_ids = _as_tuple(self.frame_ids, "frame_ids")
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)
        if self.time_unit not in {"frame", "s", "ms", "us", "ns"}:
            raise ValueError(f"unknown time_unit: {self.time_unit!r}")

    @property
    def is_new_measurement(self) -> bool:
        """Whether this frame is a new sample rather than padded repetition.

        Padding frames repeat the first real frame; counting them as new
        independent evidence is exactly the error the contract is meant to
        prevent. A missing flag means the implementation must not assume new.
        """
        if self.new_measurement is None:
            return True
        value = self.new_measurement
        if isinstance(value, bool):
            return value
        try:
            return bool(value.all())  # torch/numpy scalar-tensor compatible
        except AttributeError:
            return bool(value)


@dataclass(slots=True)
class EmbodimentContext:
    """Optional context available *at the current time step*.

    Every field pairs with an availability flag. Missing context stays missing;
    implementations must not silently substitute a default that looks like real
    feedback (for example an all-zero "executed action").
    """

    proprio: Any = None
    proprio_available: Any = None
    executed_action: Any = None
    executed_action_available: Any = None
    camera_transform: Any = None
    camera_transform_available: Any = None
    task_tokens: Any = None
    task_mask: Any = None
    hypothesis_prior: Any = None  # optional declared intervention, [B,K] nonnegative mass
    hypothesis_prior_available: Any = None  # [B] bool; never inferred from evaluator truth

    @property
    def has_proprio(self) -> bool:
        return self.proprio is not None and self.proprio_available is not False

    @property
    def has_executed_action(self) -> bool:
        return self.executed_action is not None and self.executed_action_available is not False


@dataclass(slots=True)
class ComputeBudget:
    """Deterministic compute bound for one :meth:`Encoder.step` call.

    ``max_updates`` is the number of internal update cycles, ``max_history`` the
    number of retained measurements. ``deadline_ms`` is an optional wall-clock
    soft budget: the first round reports deterministic update counts instead of
    claiming a hard real-time guarantee.
    """

    max_updates: int = 4
    max_history: int = 4
    deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_updates < 1:
            raise ValueError("max_updates must be >= 1")
        if self.max_history < 1:
            raise ValueError("max_history must be >= 1")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive when set")


@dataclass(slots=True)
class CostRecord:
    """Measured cost of one step. Unmeasured quantities stay ``None``."""

    updates: int = 0
    history_len: int = 0
    hypotheses: int = 0
    updated_fraction: float | None = None
    state_bytes: int | None = None
    wall_ms: float | None = None
    peak_bytes: int | None = None
    notes: dict = field(default_factory=dict)


@dataclass(slots=True)
class Validity:
    """Which parts of the output are supported, missing, saturated or unsolved."""

    valid_mask: Any = None
    missing_mask: Any = None
    saturated_mask: Any = None
    unresolved_ambiguity: Any = None
    solved: bool = True
    reason: str = ""


@dataclass(slots=True)
class EncoderOutput:
    """Result of one :meth:`Encoder.step` call."""

    representation: Any
    state: Any
    validity: Validity
    diagnostics: dict = field(default_factory=dict)
    cost: CostRecord = field(default_factory=CostRecord)


@dataclass(slots=True)
class VisualTokens:
    """Backend-facing readout: ``values`` is ``[B, L, descriptor_dim]``.

    ``positions`` records each token's actual image coordinate and support
    scale; ``camera_ids`` and ``valid`` keep the camera identity and mask
    explicit so a backend cannot silently merge tokens across cameras.
    """

    values: Any
    mask: Any = None
    camera_ids: tuple = ()
    positions: Any = None
    scale: Any = None
    validity: Any = None
    uncertainty: Any = None

    def __post_init__(self) -> None:
        self.camera_ids = _as_tuple(self.camera_ids, "camera_ids")


@dataclass(slots=True)
class CameraSpec:
    """Static description of one camera input stream."""

    camera_id: str
    height: int
    width: int
    channels: int = 1
    input_mode: str = "bayer_cfa"
    cfa_pattern: str = ""
    black_level: float = 0.0
    white_level: float = 1.0

    def __post_init__(self) -> None:
        if self.input_mode not in {"bayer_cfa", "rawvla_rgb3"}:
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")
        if self.input_mode == "bayer_cfa" and self.cfa_pattern not in {
            "RGGB",
            "BGGR",
            "GRBG",
            "GBRG",
        }:
            raise ValueError(f"bayer_cfa requires a CFA pattern, got {self.cfa_pattern!r}")

    @property
    def pixels(self) -> int:
        return self.height * self.width


@runtime_checkable
class Encoder(Protocol):
    """Lifecycle a persistent encoder must expose.

    Training calls :meth:`step` frame by frame with exactly the same method as
    online inference, detaching the state at truncated-BPTT boundaries. The
    same frame must not accumulate its own likelihood twice: internal cycles
    recompute the full score on a frozen measurement set.
    """

    def init_state(self, batch_size: int, camera_spec: CameraSpec, device: Any) -> Any: ...

    def step(
        self,
        frame: MeasurementFrame,
        context: EmbodimentContext,
        state: Any,
        budget: ComputeBudget,
    ) -> EncoderOutput: ...

    def reset(self, state: Any, batch_mask: Any) -> Any: ...

    def detach_state(self, state: Any) -> Any: ...


@runtime_checkable
class Readout(Protocol):
    """Turns a candidate representation into π0.5-ready visual tokens."""

    def __call__(self, output: EncoderOutput, token_budget: int) -> VisualTokens: ...


def state_tensor_bytes(state: Any) -> int:
    """Best-effort state size in bytes, for cost reporting.

    Returns 0 for states whose tensors cannot be located; callers must report
    the corresponding field as unavailable rather than as a measured zero.
    """
    total = 0
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (str, bytes, int, float, bool)):
            return
        element_size = getattr(value, "element_size", None)
        numel = getattr(value, "numel", None)
        if callable(element_size) and callable(numel):
            nonlocal total
            total += int(element_size()) * int(numel())
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for item in dataclasses.fields(value):
                visit(getattr(value, item.name))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(state)
    return total

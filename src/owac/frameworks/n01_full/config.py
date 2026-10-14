"""N01 v1.0 foundation configuration: what the route is allowed to assume.

Seven explicit sections mirror ``docs/prompts/n01_v1_0/P0_FOUNDATION.md``
section 2: input identity, basis, history, replay, correspondence, content and
training budget.  Defaults follow
``docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md``: the data window is
``[B,T,C,H,W]`` ordered oldest-to-newest, ``C=4`` is CFA and ``C=3`` is
``rawvla_rgb3``, ``history`` defaults to 8 unique frames and the exact replay
window to 32 unique frames.

Serialization is strict on purpose.  ``to_dict`` emits plain JSON (tuples
become lists) and ``from_dict`` round-trips it, but an unknown field at any
nesting level raises instead of being dropped, because a silently ignored
field is a configuration that does not match the run that claims it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Self, get_origin, get_type_hints

__all__ = [
    "CFA_PATTERNS",
    "INPUT_MODES",
    "BASIS_KINDS",
    "PRIOR_KINDS",
    "REPLAY_MODES",
    "EXACT_WINDOW_CHOICES",
    "BEYOND_EXACT_POLICIES",
    "MAPPING_KINDS",
    "INTERPOLATIONS",
    "BACKBONES",
    "CONTENT_PATHS",
    "TRAINABLE_SCOPES",
    "InputIdentityConfig",
    "BasisConfig",
    "HistoryConfig",
    "ReplayConfig",
    "CorrespondenceConfig",
    "ContentConfig",
    "TrainingBudgetConfig",
    "N01FullConfig",
    "load_config",
]

INPUT_MODES = ("bayer_cfa", "synthetic_cfa", "rawvla_rgb3")
CFA_PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")
BASIS_KINDS = ("phase_local_constant", "phase_local_affine", "directional_per_phase")
PRIOR_KINDS = ("none", "gaussian")
REPLAY_MODES = ("fixed_decision_replay", "counterfactual_replay")
EXACT_WINDOW_CHOICES = (8, 16, 32)
BEYOND_EXACT_POLICIES = ("invalidate_summary_and_rebuild",)
MAPPING_KINDS = ("identity", "sea_raft")
INTERPOLATIONS = ("nearest", "bilinear", "bicubic")
BACKBONES = ("dbsr", "bsrt")
CONTENT_PATHS = ("source_features", "domain_content")
TRAINABLE_SCOPES = ("frozen_backbone", "tuned_backbone")

#: RGB is an independent system baseline and a preview, never a main-path input.
FORBIDDEN_INPUT_MODES = ("rgb", "srgb", "xyz")


@dataclass
class InputIdentityConfig:
    """What one measurement is, in which unit and in which window.

    ``input_mode`` names one of the four data identities of the plan: real
    ``bayer_cfa``, ``synthetic_cfa`` and the author ``rawvla_rgb3``
    pseudo-RAW.  RGB modes are refused here rather than ignored, because the
    main encoder chain never consumes an RGB/sRGB/XYZ intermediate.
    """

    dataset_id: str = "rawvla_bench"
    input_mode: str = "rawvla_rgb3"
    channels: int = 3
    cfa_pattern: str | None = None
    value_units: str = "pseudo_raw_hdr"
    coordinate_units: str = "sensor_pixel"
    batch_layout: str = "BTCHW"
    time_order: str = "oldest_first"

    @property
    def is_cfa(self) -> bool:
        """Whether the identity carries real or synthetic CFA phases."""
        return self.input_mode in ("bayer_cfa", "synthetic_cfa")

    @classmethod
    def bayer_cfa(cls, dataset_id: str, cfa_pattern: str = "RGGB", **overrides: Any) -> Self:
        """Real or synthetic CFA identity: four phases, never a fourth fake measurement."""
        payload: dict[str, Any] = {
            "dataset_id": dataset_id,
            "input_mode": "bayer_cfa",
            "channels": 4,
            "cfa_pattern": cfa_pattern,
            "value_units": "raw_integer_counts",
        }
        payload.update(overrides)
        return cls(**payload).validate()

    def validate(self) -> Self:
        if self.input_mode in FORBIDDEN_INPUT_MODES:
            raise ValueError(
                f"input_mode {self.input_mode!r} is an RGB-space intermediate; the main "
                "N01 chain consumes RAW measurements only (RGB is baseline/preview)"
            )
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {self.input_mode!r}; expected {INPUT_MODES}")
        if not self.dataset_id:
            raise ValueError("InputIdentityConfig.dataset_id must be non-empty")
        if self.batch_layout != "BTCHW":
            raise ValueError(
                f"batch_layout must be 'BTCHW' ([B,T,C,H,W]), got {self.batch_layout!r}"
            )
        if self.time_order != "oldest_first":
            raise ValueError(
                "time_order must be 'oldest_first' ([B,T,C,H,W] runs old to new); the "
                "newest-first fusion entry is adapted later by the named adapter"
            )
        if self.is_cfa:
            if self.channels != 4:
                raise ValueError(f"{self.input_mode} carries four CFA phases (C=4)")
            if self.cfa_pattern not in CFA_PATTERNS:
                raise ValueError(
                    f"Unknown CFA pattern {self.cfa_pattern!r}; expected {CFA_PATTERNS}"
                )
        else:
            if self.channels != 3:
                raise ValueError("rawvla_rgb3 carries three channels (C=3)")
            if self.cfa_pattern is not None:
                raise ValueError(
                    "rawvla_rgb3 has no CFA phase; do not attach a pattern or fabricate a "
                    "fourth measurement"
                )
        if not self.value_units:
            raise ValueError("InputIdentityConfig.value_units must name the raw unit")
        if not self.coordinate_units:
            raise ValueError("InputIdentityConfig.coordinate_units must name the coordinate unit")
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class BasisConfig:
    """The structural basis of a support domain and how a prior is counted.

    ``phase_local_constant`` fits ``phi=[1]`` and ``phase_local_affine`` fits
    ``phi=[1,x,y]`` per CFA phase; the phases are never averaged first.  A
    declared prior is counted once for the whole estimate
    (``prior_shared_across_domains``), and rank deficiency is reported through
    the identifiable mask rather than hidden in a ridge term.
    """

    kind: str = "phase_local_affine"
    per_phase: bool = True
    neighbourhood_radius: int = 1
    prior_kind: str = "none"
    prior_id: str | None = None
    prior_shared_across_domains: bool = True
    rank_deficiency_policy: str = "identifiable_mask"

    def validate(self) -> Self:
        if self.kind not in BASIS_KINDS:
            raise ValueError(f"Unknown basis kind {self.kind!r}; expected {BASIS_KINDS}")
        if not self.per_phase:
            raise ValueError("the first reference fits each CFA phase independently")
        if self.neighbourhood_radius < 0:
            raise ValueError("BasisConfig.neighbourhood_radius must be non-negative")
        if self.prior_kind not in PRIOR_KINDS:
            raise ValueError(f"Unknown prior_kind {self.prior_kind!r}; expected {PRIOR_KINDS}")
        if self.prior_kind == "none":
            if self.prior_id is not None:
                raise ValueError("prior_id must be None when prior_kind is 'none'")
        else:
            if not self.prior_id:
                raise ValueError(f"prior_kind {self.prior_kind!r} requires an explicit prior_id")
            if not self.prior_shared_across_domains:
                raise ValueError(
                    "a prior is counted once per estimate, not per merged domain; "
                    "prior_shared_across_domains cannot be disabled"
                )
        if self.rank_deficiency_policy != "identifiable_mask":
            raise ValueError(
                "a rank-deficient design must return an identifiable mask (optionally with "
                "unavailable covariance), not a regularized point estimate"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class HistoryConfig:
    """How much unique history is retained and which windows are compared."""

    max_history: int = 8
    comparison_windows: tuple[int, ...] = (8, 16, 32)
    summary_keeps_provenance: bool = True

    def validate(self) -> Self:
        if self.max_history < 1:
            raise ValueError("HistoryConfig.max_history must be at least one unique frame")
        if not self.comparison_windows:
            raise ValueError("HistoryConfig.comparison_windows must not be empty")
        if tuple(sorted(set(self.comparison_windows))) != tuple(self.comparison_windows):
            raise ValueError("comparison_windows must be strictly increasing unique windows")
        if min(self.comparison_windows) < 1:
            raise ValueError("comparison window lengths must be positive")
        if not self.summary_keeps_provenance:
            raise ValueError(
                "information older than the exact window survives only as a provenance-"
                "carrying summary; it cannot be kept untraceable"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class ReplayConfig:
    """The two replay interfaces and the exact retraction window.

    ``fixed_decision_replay`` keeps motion/support decisions fixed and checks
    that retracted content no longer affects the readout.
    ``counterfactual_replay`` deletes the source from the input stream and
    recomputes everything; only that mode may claim the result is independent
    of the source.  Both modes exist from the first version so "retraction
    succeeded" cannot stay one ambiguous metric.
    """

    modes: tuple[str, ...] = REPLAY_MODES
    default_mode: str = "fixed_decision_replay"
    exact_window_unique_frames: int = 32
    beyond_exact_policy: str = "invalidate_summary_and_rebuild"
    require_unique_frames: bool = True
    drop_derived_caches_on_parameter_update: bool = True

    def validate(self) -> Self:
        if set(self.modes) != set(REPLAY_MODES):
            raise ValueError(
                f"both replay interfaces are required, got {self.modes!r}; expected {REPLAY_MODES}"
            )
        if self.default_mode not in self.modes:
            raise ValueError(f"default_mode {self.default_mode!r} is not declared in modes")
        if self.exact_window_unique_frames not in EXACT_WINDOW_CHOICES:
            raise ValueError(
                "the first reference keeps an exact window of 8/16/32 unique frames, got "
                f"{self.exact_window_unique_frames!r}"
            )
        if self.beyond_exact_policy not in BEYOND_EXACT_POLICIES:
            raise ValueError(
                f"Unknown beyond_exact_policy {self.beyond_exact_policy!r}; expected "
                f"{BEYOND_EXACT_POLICIES}"
            )
        if not self.require_unique_frames:
            raise ValueError(
                "the exact window counts unique frames; duplicated packets must observe once"
            )
        if not self.drop_derived_caches_on_parameter_update:
            raise ValueError(
                "a parameter update retires derived caches produced under the old version"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class CorrespondenceConfig:
    """How a reference position pulls from a historical source frame.

    ``source_xy`` is always a pull coordinate (reference position -> historical
    source); the push convention is refused here.  CFA modes must preserve the
    sampled phase, and forward/backward consistency may look at the current and
    the past two frames only, never at a future frame.
    """

    mapping_kind: str = "identity"
    mapping_version: str = "n01-full-v1"
    coordinate_convention: str = "pull"
    interpolation: str = "bilinear"
    preserve_cfa_phase: bool = True
    input_channels: int | None = None
    forward_backward_check: bool = True
    backward_window_frames: int = 2

    def validate(self) -> Self:
        if self.mapping_kind not in MAPPING_KINDS:
            raise ValueError(
                f"Unknown mapping_kind {self.mapping_kind!r}; expected {MAPPING_KINDS}"
            )
        if not self.mapping_version:
            raise ValueError("CorrespondenceConfig.mapping_version must be declared")
        if self.coordinate_convention != "pull":
            raise ValueError(
                "source_xy must be a pull coordinate (reference position -> historical "
                f"source); {self.coordinate_convention!r} is not this convention"
            )
        if self.interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"Unknown interpolation {self.interpolation!r}; expected {INTERPOLATIONS}"
            )
        if self.input_channels is not None and self.input_channels < 1:
            raise ValueError("CorrespondenceConfig.input_channels must be positive")
        if not 0 <= self.backward_window_frames <= 2:
            raise ValueError(
                "forward/backward flow consistency uses the current and the past two frames "
                "only; future frames are not available"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class ContentConfig:
    """The mature content backbone and the explicit bridge dimensions.

    The first version fixes the full-size DBSR (64 channels, 9 residual blocks,
    512 features) and its readout (256 tokens of width 2048).  The bridge is
    constructed with the explicit ``bridge_in_dim=512`` and
    ``bridge_width=2048``; the backend default that would set
    ``in_dim=readout_width`` is never used.  ``fusion_time_order`` names the
    newest-first DBSR fusion entry, and ``time_order_adapter`` names the single
    function allowed to convert the ``[B,T,C,H,W]`` oldest-first window.
    """

    backbone: str = "dbsr"
    channels: int = 64
    num_blocks: int = 9
    feature_width: int = 512
    readout_tokens: int = 256
    readout_width: int = 2048
    bridge_in_dim: int = 512
    bridge_width: int = 2048
    path: str = "source_features"
    trainable_scope: str = "frozen_backbone"
    fusion_time_order: str = "newest_first"
    time_order_adapter: str = "oldest_first_to_newest_first"

    def validate(self) -> Self:
        if self.backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {self.backbone!r}; expected {BACKBONES}")
        if self.backbone == "dbsr":
            expected = {"channels": 64, "num_blocks": 9, "feature_width": 512}
            wrong = {
                name: getattr(self, name)
                for name, value in expected.items()
                if value != getattr(self, name)
            }
            if wrong:
                raise ValueError(
                    f"the first version uses the full-size DBSR {expected}; got {wrong}"
                )
            if self.readout_tokens != 256 or self.readout_width != 2048:
                raise ValueError(
                    "the first DBSR readout returns 256 tokens of width 2048 "
                    f"(got {self.readout_tokens} x {self.readout_width})"
                )
        for name in ("channels", "num_blocks", "feature_width", "readout_tokens", "readout_width"):
            if getattr(self, name) < 1:
                raise ValueError(f"ContentConfig.{name} must be positive")
        if self.bridge_in_dim != self.feature_width:
            raise ValueError(
                "the bridge input width must equal the content feature width "
                f"({self.bridge_in_dim} != {self.feature_width}); do not rely on the "
                "backend default in_dim=readout_width"
            )
        if self.bridge_width != self.readout_width:
            raise ValueError(
                f"bridge_width {self.bridge_width} must equal readout_width {self.readout_width}"
            )
        if self.path not in CONTENT_PATHS:
            raise ValueError(f"Unknown content path {self.path!r}; expected {CONTENT_PATHS}")
        if self.trainable_scope not in TRAINABLE_SCOPES:
            raise ValueError(
                f"Unknown trainable_scope {self.trainable_scope!r}; expected {TRAINABLE_SCOPES}"
            )
        if self.fusion_time_order != "newest_first":
            raise ValueError(
                "the DBSR fusion entry consumes newest first; keep the declaration explicit"
            )
        if not self.time_order_adapter:
            raise ValueError(
                "the oldest-to-newest conversion must live in one named adapter function"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class TrainingBudgetConfig:
    """Resource candidates, validation cadence and the seed streams.

    Every number here is a planning candidate from the plan, not evidence of
    sufficient training.  Model seeds and data seeds are separate streams, and
    at least three model seeds are declared for the core comparison.
    """

    mechanism_updates: int = 20_000
    mechanism_updates_max: int = 100_000
    motion_updates: int = 100_000
    motion_updates_max: int = 300_000
    action_updates: int = 30_000
    action_updates_max: int = 100_000
    validation_interval: int = 1_000
    plateau_checks: int = 8
    model_seeds: tuple[int, ...] = (101, 211, 307)
    data_seeds: tuple[int, ...] = ()
    data_seed_namespace: str = "n01-full-v1"

    def validate(self) -> Self:
        for name in (
            "mechanism_updates",
            "mechanism_updates_max",
            "motion_updates",
            "motion_updates_max",
            "action_updates",
            "action_updates_max",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"TrainingBudgetConfig.{name} must be positive")
        for base in ("mechanism", "motion", "action"):
            planned = getattr(self, f"{base}_updates")
            ceiling = getattr(self, f"{base}_updates_max")
            if planned > ceiling:
                raise ValueError(f"{base}_updates {planned} exceeds its max {ceiling}")
        if self.validation_interval < 1:
            raise ValueError("TrainingBudgetConfig.validation_interval must be positive")
        if self.plateau_checks != 8:
            raise ValueError(
                "validation_plateau requires 8 consecutive improvements below the locked "
                f"delta_val; got plateau_checks={self.plateau_checks}"
            )
        if len(self.model_seeds) < 3:
            raise ValueError("the core comparison declares at least three model seeds")
        if len(set(self.model_seeds)) != len(self.model_seeds):
            raise ValueError("model_seeds must be distinct")
        if len(set(self.data_seeds)) != len(self.data_seeds):
            raise ValueError("data_seeds must be distinct")
        shared = set(self.model_seeds) & set(self.data_seeds)
        if shared:
            raise ValueError(
                f"data seeds and model seeds are separate streams; {sorted(shared)} appear in both"
            )
        if not self.data_seed_namespace:
            raise ValueError("TrainingBudgetConfig.data_seed_namespace must be declared")
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()


@dataclass
class N01FullConfig:
    """The complete P0 configuration of the N01 v1.0 route."""

    config_version: str = "n01-full-v1"
    input: InputIdentityConfig = field(default_factory=InputIdentityConfig)
    basis: BasisConfig = field(default_factory=BasisConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    correspondence: CorrespondenceConfig = field(default_factory=CorrespondenceConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    budget: TrainingBudgetConfig = field(default_factory=TrainingBudgetConfig)

    def __post_init__(self) -> None:
        # Constructing the top-level object already means declaring a valid one.
        self.validate()

    def validate(self) -> Self:
        if not self.config_version:
            raise ValueError("N01FullConfig.config_version must be declared")
        self.input.validate()
        self.basis.validate()
        self.history.validate()
        self.replay.validate()
        self.correspondence.validate()
        self.content.validate()
        self.budget.validate()
        exact = self.replay.exact_window_unique_frames
        if self.history.max_history > exact:
            raise ValueError(
                f"max_history {self.history.max_history} exceeds the exact replay window "
                f"{exact}; the first version cannot promise exact retraction beyond it"
            )
        if max(self.history.comparison_windows) > exact:
            raise ValueError(
                f"comparison window {max(self.history.comparison_windows)} exceeds the exact "
                f"replay window {exact}"
            )
        if self.input.is_cfa and not self.correspondence.preserve_cfa_phase:
            raise ValueError(
                f"{self.input.input_mode} keeps the sampled CFA phase; preserve_cfa_phase "
                "cannot be disabled"
            )
        channels = self.correspondence.input_channels
        if channels is not None and channels % self.input.channels:
            raise ValueError(
                f"correspondence input_channels {channels} is not a whole number of "
                f"C={self.input.channels} measurement channels"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return _from_plain(cls, data).validate()

    @classmethod
    def from_json(cls, text: str) -> Self:
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise TypeError("an N01FullConfig JSON document must be an object")
        return cls.from_dict(payload)

    def fingerprint(self) -> str:
        """Stable hash of the frozen configuration, for receipts and manifests."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(source: str | Path | Mapping[str, Any]) -> N01FullConfig:
    """Load a strict config from a JSON file path or an already parsed mapping."""
    if isinstance(source, Mapping):
        return N01FullConfig.from_dict(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"N01 full config not found: {path}")
    return N01FullConfig.from_dict(json.loads(path.read_text()))


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _to_plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    return value


def _from_plain(cls: type, data: Mapping[str, Any]) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    declared = [item.name for item in fields(cls)]
    unknown = sorted(set(data) - set(declared))
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    kwargs: dict[str, Any] = {}
    for name in declared:
        if name not in data:
            continue
        value = data[name]
        hint = hints[name]
        if is_dataclass(hint) and isinstance(value, Mapping):
            nested = getattr(hint, "from_dict", None)
            value = nested(value) if nested is not None else _from_plain(hint, value)
        elif get_origin(hint) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)

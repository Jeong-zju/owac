"""Configuration for the N02 measurement-constrained recurrent completion field.

One config object drives the tiny/small/medium ablations. Every knob that the
mechanism claim depends on is explicit here, because the revision rule must be
switchable off (``disable_dependency_propagation``, ``disable_contradiction``)
to serve as its own ablation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MeasurementConfig:
    """Measurement-space knobs: input mode, noise model, robust likelihood."""

    input_mode: str = "rawvla_rgb3"
    cfa_pattern: str = "RGGB"
    black_level: float = 0.0
    white_level: float = 255.0
    robust: str = "huber"  # huber | gaussian
    huber_delta: float = 3.0
    variance_floor: float = 1.0e-6
    shot_scale: float | None = None
    read_scale: float | None = None
    scale_estimated: bool = False
    predict_variance: bool = True
    head_width: int = 32
    head_depth: int = 2
    geometry_dim: int = 8
    saturation_is_observation: bool = False

    def __post_init__(self) -> None:
        if self.input_mode not in {"bayer_cfa", "rawvla_rgb3"}:
            raise ValueError(f"unknown input_mode: {self.input_mode!r}")
        if self.robust not in {"huber", "gaussian"}:
            raise ValueError(f"unknown robust mode: {self.robust!r}")


@dataclass(slots=True)
class RevisionConfig:
    """Contradiction localisation, dependency propagation and local update budget."""

    contradiction_enabled: bool = True
    contradiction_threshold: float = 2.5
    relative_margin: float = 1.5
    strong_contradiction_threshold: float = 4.0
    residual_ema: float = 0.5
    propagation_enabled: bool = True
    propagation_hops: int = 1
    propagation_decay: float = 0.5
    propagate_priority: bool = True
    reset_contradicted: bool = True
    reset_fraction: float = 1.0
    downweight_contradicted: float = 0.5
    proposal_enabled: bool = True
    proposal_temperature: float = 1.0
    update_scale: float = 1.0
    cold_start_threshold: float = 0.25
    tolerance: float = 1.0e-4
    min_iterations: int = 1
    write_dependency_edges: bool = True

    def __post_init__(self) -> None:
        if self.propagation_hops < 0:
            raise ValueError("propagation_hops must be >= 0")
        if not 0.0 < self.propagation_decay <= 1.0:
            raise ValueError("propagation_decay must be in (0, 1]")
        if not 0.0 < self.reset_fraction <= 1.0:
            raise ValueError("reset_fraction must be in (0, 1]")
        if not 0.0 <= self.downweight_contradicted <= 1.0:
            raise ValueError("downweight_contradicted must be in [0, 1]")


@dataclass(slots=True)
class LossWeights:
    """Tiny-stage objective weights, with normalisation stated per term.

    ``measurement``/``geometry_set``/``revision``/``diversity`` are summed over
    their own valid-sample counts and divided by that count, so each reported
    term is a per-observation mean before weighting. Set a weight to ``0.0`` to
    ablate the term; the term is still reported as an explicit zero.
    """

    measurement: float = 1.0
    geometry_set: float = 1.0
    revision: float = 0.1
    diversity: float = 0.01
    mixture: float = 0.0
    collapse_penalty: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class N02Config:
    """Full N02 configuration. ``tiny`` is the CPU mechanism configuration."""

    name: str = "tiny"
    K: int = 4
    D: int = 32
    cycles: int = 4
    history: int = 4
    temperature: float = 1.0
    beta_prior: float = 1.0
    beta_relation: float = 0.0
    beta_prior_bounds: tuple[float, float] = (0.05, 4.0)
    stem_width: int = 32
    stem_layers: int = 2
    cell_kind: str = "convgru"  # convgru | residual
    cell_kernel: int = 3
    readout_mode: str = "map"  # map | tokens
    tokens_per_camera: int = 256
    token_dim: int = 2048
    prior_mode: str = "learned"  # learned | fixed_odds
    geometry_dim: int = 8
    seed: int = 0
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    revision: RevisionConfig = field(default_factory=RevisionConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError("K must be >= 1")
        if self.D < 1:
            raise ValueError("D must be >= 1")
        if self.cycles < 1:
            raise ValueError("cycles must be >= 1")
        if self.history < 1:
            raise ValueError("history must be >= 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        low, high = self.beta_prior_bounds
        if not 0.0 <= low < high:
            raise ValueError("beta_prior_bounds must satisfy 0 <= low < high")
        if self.cell_kind not in {"convgru", "residual"}:
            raise ValueError(f"unknown cell_kind: {self.cell_kind!r}")
        if self.readout_mode not in {"map", "tokens"}:
            raise ValueError(f"unknown readout_mode: {self.readout_mode!r}")
        if self.prior_mode not in {"learned", "fixed_odds"}:
            raise ValueError(f"unknown prior_mode: {self.prior_mode!r}")

    @property
    def hypothesis_grid(self) -> tuple[int, int]:
        return (self.K, self.D)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> N02Config:
        data = dict(data)
        nested = {
            "measurement": MeasurementConfig,
            "revision": RevisionConfig,
            "loss_weights": LossWeights,
        }
        kwargs: dict[str, Any] = {}
        allowed = {item.name for item in fields(cls)}
        for key, value in data.items():
            if key not in allowed:
                raise ValueError(f"unknown N02Config key: {key!r}")
            if key in nested:
                kwargs[key] = nested[key](**value)
            elif key == "beta_prior_bounds":
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Path) -> N02Config:
        return cls.from_dict(json.loads(Path(path).read_text()))


PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {"K": 4, "D": 32, "cycles": 4, "history": 4, "stem_width": 32},
    "small": {"K": 4, "D": 64, "cycles": 6, "history": 4, "stem_width": 48},
    "medium": {"K": 8, "D": 96, "cycles": 8, "history": 4, "stem_width": 64},
}


def preset(name: str) -> N02Config:
    """Named scale presets from the implementation prompt (tiny/small/medium)."""
    if name not in PRESETS:
        raise ValueError(f"unknown N02 preset: {name!r}; available: {sorted(PRESETS)}")
    return N02Config(name=name, **PRESETS[name])


def load_config(path: Path | str | None = None, preset_name: str | None = None) -> N02Config:
    """Load a JSON config, a named preset, or the ``tiny`` default."""
    if path is not None:
        return N02Config.from_json(Path(path))
    return preset(preset_name or "tiny")

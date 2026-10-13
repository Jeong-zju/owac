"""N01 configuration and scale presets.

The tiny/small/medium presets are the starting parameters named in the N01
implementation prompt; they are a runnable starting point, not a tuned training
protocol.  Width, spatial scale, longest history and block count stay
independently adjustable so that scale trends can be measured rather than
assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["N01Config", "load_config", "config_fingerprint", "SCALE_PRESETS"]

_DEFAULT_LOSS_WEIGHTS = {
    "location": 1.0,
    "measurement_prediction": 0.1,
    "budget": 0.0,
}


@dataclass
class N01Config:
    """Validated configuration for the N01 encoder and its mechanism probes."""

    input_mode: str = "rawvla_rgb3"
    channels: int = 3
    planes: int = 1
    cfa_pattern: str = "RGGB"
    width: int = 32
    num_blocks: int = 2
    neighbourhood_radius: int = 2
    spatial_radii: tuple[int, ...] = (0, 1, 2)
    time_depths: tuple[int, ...] = (1, 4, 8)
    max_history: int = 8
    support_power: float = 2.0
    support_floor: float = 0.05
    merge_threshold: float = 0.55
    split_threshold: float = 0.35
    edge_temperature: float = 0.10
    edge_alpha: float = 0.35
    edge_beta: float = 0.35
    edge_forget: float = 0.10
    learnable_gate: bool = True
    gate_hidden: int = 8
    alignment: str = "none"
    correspondence_window: int = 3
    correspondence_width: int = 16
    fractional_alignment: bool = False
    soft_argmax: bool = True
    soft_argmax_temperature: float = 0.05
    selection: str = "analytic"
    # gate_only preserves the original statistics-conditioned update exactly.
    # current_control matches integrated's extra capacity with current RAW only.
    content_mode: str = "gate_only"
    # Experimental source-wise innovation risk, disabled for existing checkpoints.
    source_reliability: str = "none"
    source_reliability_floor: float = 0.01
    selection_use_learned: bool = False
    selector_temperature: float = 0.1
    cost_weights: dict[str, float] = field(
        # Starting values only: the proposal fixes the cost *form*, and these
        # weights are to be selected on a validation split.  The noise term is
        # deliberately the strongest so that a still scene can afford a longer
        # integration while a moving one cannot.
        default_factory=lambda: {
            "noise": 2.0,
            "edge": 0.5,
            "age": 0.15,
            "motion": 0.5,
        }
    )
    budget_conditioning: bool = False
    trainable_support: bool = False
    aux_measurement_prediction: bool = True
    readout_tokens: int = 256
    readout_width: int = 2048
    require_unique_frames: bool = True
    max_domain_pixels: int = 0  # 0 -> the (2r+1)^2 window implied by the radius
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_LOSS_WEIGHTS))
    noise_shot: float = 0.0
    noise_read: float = 0.0
    deadline_ms: float | None = None
    task_precision: float | None = None

    # -- validation ---------------------------------------------------------
    def validate(self) -> N01Config:
        if self.input_mode not in {"rawvla_rgb3", "bayer_cfa", "sequence1d"}:
            raise ValueError(f"Unknown input_mode {self.input_mode!r}")
        if self.planes not in (1, 4):
            raise ValueError("planes must be 1 (three-channel measurement) or 4 (2x2 CFA)")
        if self.planes == 4 and self.channels != self.planes:
            raise ValueError("bayer_cfa expects one measured channel per CFA plane")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.width < 1 or self.num_blocks < 1:
            raise ValueError("width and num_blocks must be positive")
        if self.max_history < 1:
            raise ValueError("max_history must be positive")
        if not self.spatial_radii or min(self.spatial_radii) < 0:
            raise ValueError("spatial_radii must be non-negative")
        if max(self.spatial_radii) > self.neighbourhood_radius:
            raise ValueError("neighbourhood_radius must cover every spatial radius")
        if not self.time_depths or min(self.time_depths) < 1:
            raise ValueError("time_depths must be >= 1")
        if max(self.time_depths) > self.max_history:
            raise ValueError("time_depths must not exceed max_history")
        if self.require_unique_frames and self.max_history > 8:
            raise ValueError("the first reference implementation keeps at most 8 unique frames")
        if not self.split_threshold < self.merge_threshold:
            raise ValueError("merge_threshold must exceed split_threshold (hysteresis band)")
        if self.edge_temperature <= 0:
            raise ValueError("edge_temperature must be positive")
        if self.soft_argmax_temperature <= 0:
            raise ValueError("soft_argmax_temperature must be positive")
        if self.selector_temperature <= 0:
            raise ValueError("selector_temperature must be positive")
        if not 0.0 <= self.support_floor < 1.0:
            raise ValueError("support_floor must lie in [0, 1)")
        if self.alignment not in {"none", "fixed", "candidate"}:
            raise ValueError(f"Unknown alignment {self.alignment!r}")
        if self.selection not in {"analytic", "learned"}:
            raise ValueError(f"Unknown selection {self.selection!r}")
        if self.content_mode not in {"gate_only", "integrated", "current_control"}:
            raise ValueError(f"Unknown content_mode {self.content_mode!r}")
        if self.source_reliability not in {"none", "innovation_risk"}:
            raise ValueError(f"Unknown source_reliability {self.source_reliability!r}")
        if not 0 <= self.source_reliability_floor < 1:
            raise ValueError("source_reliability_floor must lie in [0, 1)")
        if self.input_mode == "sequence1d" and self.source_reliability != "none":
            raise ValueError("sequence1d does not yet support source_reliability")
        if self.input_mode == "sequence1d" and self.content_mode != "gate_only":
            raise ValueError("sequence1d currently supports only content_mode='gate_only'")
        if self.planes == 4 and self.cfa_pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            raise ValueError("Unknown CFA pattern")
        missing = {"location", "measurement_prediction", "budget"} - set(self.loss_weights)
        if missing:
            raise ValueError(f"loss_weights is missing {sorted(missing)}")
        for key, value in self.cost_weights.items():
            if value < 0:
                raise ValueError(f"cost weight {key!r} must be non-negative")
        return self

    # -- presets ------------------------------------------------------------
    @classmethod
    def tiny(cls, **overrides: Any) -> N01Config:
        return cls(**overrides).validate()

    @classmethod
    def small(cls, **overrides: Any) -> N01Config:
        base = {"width": 64, "num_blocks": 3, "max_history": 8}
        base.update(overrides)
        return cls(**base).validate()

    @classmethod
    def medium(cls, **overrides: Any) -> N01Config:
        base = {"width": 128, "num_blocks": 4, "max_history": 8}
        base.update(overrides)
        return cls(**base).validate()

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["spatial_radii"] = list(self.spatial_radii)
        data["time_depths"] = list(self.time_depths)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> N01Config:
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"Unknown N01Config fields: {unknown}")
        payload = dict(data)
        for key in ("spatial_radii", "time_depths"):
            if key in payload:
                payload[key] = tuple(int(value) for value in payload[key])
        return cls(**payload).validate()

    def replace(self, **overrides: Any) -> N01Config:
        payload = self.to_dict()
        payload.update(overrides)
        return N01Config.from_dict(payload)

    # -- derived quantities -------------------------------------------------
    @property
    def window(self) -> int:
        """Spatial window side covering the largest candidate radius."""
        return 2 * int(max(self.spatial_radii)) + 1

    @property
    def candidate_count(self) -> int:
        """Size of the candidate set the integrator actually enumerates.

        One domain per spatial radius and time depth. Offsets are original
        samples inside a domain, never additional candidate domains.
        """
        return len(self.spatial_radii) * len(self.time_depths)

    def domain_pixel_cap(self, radius: int) -> int:
        if self.max_domain_pixels:
            return min(self.max_domain_pixels, (2 * radius + 1) ** 2)
        return (2 * radius + 1) ** 2

    def signature(self) -> dict[str, Any]:
        """Fields that change the numerical mechanism (for checkpoint audits)."""
        keys = [
            "input_mode",
            "channels",
            "planes",
            "cfa_pattern",
            "width",
            "num_blocks",
            "spatial_radii",
            "time_depths",
            "max_history",
            "support_power",
            "support_floor",
            "merge_threshold",
            "split_threshold",
            "edge_temperature",
            "neighbourhood_radius",
            "alignment",
            "selection",
        ]
        # Preserve pre-content gate-only checkpoint fingerprints. Enabled modes
        # are distinct mechanisms and always carry their explicit identity.
        if self.content_mode != "gate_only":
            keys.append("content_mode")
        if self.source_reliability != "none":
            keys.extend(["source_reliability", "source_reliability_floor"])
        data = self.to_dict()
        return {key: data[key] for key in keys}


SCALE_PRESETS = {"tiny": N01Config.tiny, "small": N01Config.small, "medium": N01Config.medium}


def load_config(source: str | Path | dict[str, Any]) -> N01Config:
    """Load a config from a JSON file, a dict, or a preset name."""
    if isinstance(source, dict):
        payload = dict(source)
    elif isinstance(source, (str, Path)):
        text = str(source)
        if text in SCALE_PRESETS:
            return SCALE_PRESETS[text]()
        path = Path(text)
        if not path.is_file():
            raise FileNotFoundError(f"N01 config not found: {path}")
        payload = json.loads(path.read_text())
    else:
        raise TypeError(f"Unsupported config source: {type(source)!r}")
    preset = payload.pop("preset", None)
    if preset is not None:
        if preset not in SCALE_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}")
        base = SCALE_PRESETS[preset]().to_dict()
        base.update(payload)
        payload = base
    return N01Config.from_dict(payload)


def config_fingerprint(config: N01Config) -> str:
    blob = json.dumps(config.signature(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

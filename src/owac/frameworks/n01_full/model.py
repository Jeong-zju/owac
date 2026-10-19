"""N01 v1.2 P3: the complete model and its four controlled arms.

The four arms answer two separate questions and must not be conflated:

===============  ==========  =========================================
method           input       content path
===============  ==========  =========================================
``raw_base``     legal RAW   full DBSR + fusion, no measurement support
``raw_n01``      same RAW    same + measurement support revision
``rgb_base``    ISP RGB     full DBSR + fusion, no measurement support
``rgb_n01``     same RGB    same + support revision in the RGB domain
===============  ==========  =========================================

So ``raw_n01`` versus ``raw_base`` isolates the same-input mechanism difference,
and ``raw_n01`` versus ``rgb_n01`` isolates the same-mechanism input difference.
Neither pairwise comparison alone establishes "RAW-specific ability", and this
module never scores them into a single number.

The identical structure is deliberate: all four arms share the three-channel
stem, the full DBSR width/depth, the same history length, the same token grid and
the same bridge, and support carries no learnable parameters.  A four-arm
comparison therefore runs at equal parameter count and equal backend exposure.
The RAW arms do not build an RGB intermediate image and take no RGB teacher: the
RGB arms read a separately tagged ``rendered_rgb3`` tensor produced by
:mod:`owac.raw.n01_paired_views` from the same RAW measurement.

``rawvla_rgb3`` is the author's three-channel HDR-derived / pseudo-RAW view, not a
native Bayer CFA.  The CFA entry stays available through
:data:`DbsrAdapterConfig` and is unaffected by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from owac.core.encoder_contract import EncoderOutput, MeasurementFrame, VisualTokens
from owac.frameworks.n01_full.adapters import (
    THREE_CHANNEL_INPUT_MODES,
    DbsrAdapterConfig,
    DbsrFeatureAdapter,
    FullState,
)
from owac.frameworks.n01_full.learned_motion import DomainWindowMotion, MotionConfig
from owac.frameworks.n01_full.observation import ObservationConfig, prepare_frame
from owac.frameworks.n01_full.support import SpatialSupport, SupportConfig
from owac.frameworks.n01_full.training_plan import TrainingPlan
from owac.frameworks.n01_full.window_support import WindowGLSSupport

__all__ = [
    "ARM_METHODS",
    "ArmSpec",
    "ARMS",
    "DEFAULT_SEED",
    "PLATFORMS",
    "N01V12Config",
    "N01V12Model",
    "arm_matrix",
    "default_camera_order",
]


DEFAULT_SEED = 101
PLATFORMS = ("libero", "robotwin2")
ARM_METHODS = ("raw_base", "raw_n01", "rgb_base", "rgb_n01")


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the four-arm protocol."""

    method: str
    input_domain: str
    display_name: str
    support_enabled: bool

    @property
    def is_raw(self) -> bool:
        return self.input_domain == "RAW"

    @property
    def is_n01(self) -> bool:
        return self.support_enabled


# Support-config keys that record provenance rather than behaviour.  They travel in the
# configuration so a reader can trace a threshold, but they are not constructor arguments.
SUPPORT_PROVENANCE_KEYS = frozenset({"source", "calibration", "threshold_provenance"})

# The arm's declared domain decides which tensor it reads.  ``RAW`` is the real
# measurement; ``RGB`` is that same measurement passed through the frozen same-source ISP.
INPUT_MODE_FOR_DOMAIN = {"RAW": "rawvla_rgb3", "RGB": "rendered_rgb3"}

ARMS: dict[str, ArmSpec] = {
    "raw_base": ArmSpec("raw_base", "RAW", "RAW × mature content (Base)", False),
    "raw_n01": ArmSpec("raw_n01", "RAW", "RAW × measurement support (N01)", True),
    "rgb_base": ArmSpec("rgb_base", "RGB", "ISP-RGB × mature content (Base)", False),
    "rgb_n01": ArmSpec("rgb_n01", "RGB", "ISP-RGB × measurement support (N01)", True),
}


def default_camera_order(platform: str) -> tuple[str, ...]:
    """Camera order from the real loader protocol, never invented here."""

    from owac.raw.rawvla_sequence import resolve_protocol

    protocol = resolve_protocol(platform)
    return tuple(protocol.camera_id(key) for key in protocol.camera_keys)


def data_kind_for(platform: str) -> str:
    """What the stored measurement actually is, for the monitor and the report."""

    key = str(platform).lower()
    if key.startswith("robotwin"):
        # RoboTwin stores the float32 linear pseudo-RAW view at native resolution.
        return "hdr_derived"
    # LIBERO stores rint(raw10 * 255 / 1023) as uint8, so the ten-bit measurement
    # is already quantised before any P3 model sees it.
    return "libero_dequantized"


@dataclass(frozen=True)
class N01V12Config:
    """Effective configuration of one arm's training run."""

    arm: str = "raw_n01"
    platform: str = "libero"
    seed: int = DEFAULT_SEED
    window_frames: int = 32
    feature_width: int = 512
    stem_width: int = 64
    residual_blocks: int = 9
    fusion: str = "dbsr_reference"
    pool_grid: int = 16
    tokens_per_camera: int = 256
    readout_width: int = 2048
    # Empty means "derive from the arm's declared input domain": the RAW arms read the
    # real measurement, the RGB arms read the same measurement through the frozen ISP.
    # Leaving this at a fixed "rawvla_rgb3" is what made rgb_n01 and raw_n01 consume the
    # identical tensor.
    input_mode: str = ""
    camera_order: tuple[str, ...] = ()
    support: SupportConfig = field(default_factory=SupportConfig)
    motion: MotionConfig | None = None
    observation: ObservationConfig | None = None
    training: TrainingPlan | None = None
    supported_logic_version: str = "n01-v1.2-p3"

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARM_METHODS}")
        if not self.input_mode:
            # The dataclass is frozen, so the resolved default is written with
            # object.__setattr__ rather than by mutating the field.
            object.__setattr__(
                self, "input_mode", INPUT_MODE_FOR_DOMAIN[ARMS[self.arm].input_domain]
            )
        if self.input_mode not in THREE_CHANNEL_INPUT_MODES:
            raise ValueError(
                f"input_mode {self.input_mode!r} is not a three-channel mode "
                f"{THREE_CHANNEL_INPUT_MODES}"
            )
        if self.platform not in PLATFORMS:
            raise ValueError(f"unknown platform {self.platform!r}; expected {PLATFORMS}")
        if self.window_frames < 2:
            raise ValueError("window_frames must be at least 2 (reference + one source)")
        if self.pool_grid * self.pool_grid != self.tokens_per_camera:
            raise ValueError(
                f"tokens_per_camera {self.tokens_per_camera} != pool_grid^2 "
                f"{self.pool_grid * self.pool_grid}"
            )

    @property
    def spec(self) -> ArmSpec:
        return ARMS[self.arm]

    @property
    def data_kind(self) -> str:
        return data_kind_for(self.platform)

    @property
    def cameras(self) -> tuple[str, ...]:
        return self.camera_order or default_camera_order(self.platform)

    def adapter_config(self) -> DbsrAdapterConfig:
        return DbsrAdapterConfig(
            input_mode=self.input_mode,
            platform=self.platform,
            feature_width=self.feature_width,
            stem_width=self.stem_width,
            residual_blocks=self.residual_blocks,
            fusion=self.fusion,
            max_history=self.window_frames,
            pool_grid=self.pool_grid,
            readout_width=self.readout_width,
            tokens_per_camera=self.tokens_per_camera,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "method": self.arm,
            # Named so no entry point can silently fall back to the prototype factory.
            "model_factory": "owac.frameworks.n01_full.model:build_model",
            "platform": self.platform,
            "seed": self.seed,
            "input_domain": self.spec.input_domain,
            "data_kind": self.data_kind,
            "support_enabled": self.spec.support_enabled,
            "window_frames": self.window_frames,
            "feature_width": self.feature_width,
            "stem_width": self.stem_width,
            "residual_blocks": self.residual_blocks,
            "fusion": self.fusion,
            "pool_grid": self.pool_grid,
            "tokens_per_camera": self.tokens_per_camera,
            "readout_width": self.readout_width,
            "input_mode": self.input_mode,
            "camera_order": list(self.cameras),
            "support": self.support.to_dict(),
            "motion": None if self.motion is None else self.motion.to_dict(),
            "observation": None if self.observation is None else self.observation.to_dict(),
            "training": None if self.training is None else self.training.to_dict(),
            "supported_logic_version": self.supported_logic_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> N01V12Config:
        data = dict(payload)
        data.pop("method", None)
        data.pop("input_domain", None)
        # ``input_mode`` is derived from the arm's declared domain, so a stored value is not
        # an independent knob.  Configurations written before the derivation existed carry
        # ``rawvla_rgb3`` for every arm, and honouring that would quietly put the RGB arms
        # back on the RAW tensor.  Dropping it lets the derivation run; ``to_dict`` writes
        # the resolved value back, so the record stays traceable.
        data.pop("input_mode", None)
        data.pop("data_kind", None)
        data.pop("support_enabled", None)
        data.pop("model_factory", None)
        support = data.pop("support", None)
        cameras = data.pop("camera_order", None)
        if data.get("motion") is not None:
            data["motion"] = MotionConfig(**data["motion"])
        if data.get("observation") is not None:
            data["observation"] = ObservationConfig.from_dict(data["observation"])
        if data.get("training") is not None:
            data["training"] = TrainingPlan.from_dict(data["training"])
        if support is not None:
            data["support"] = SupportConfig(
                **{
                    key: (tuple(value) if key == "multiscale_radii" else value)
                    for key, value in support.items()
                    # Non-constructor keys carry provenance (upstream attribution, the
                    # record of where the thresholds came from) and are dropped here.
                    # Named once so adding a provenance field cannot break loading.
                    if key not in SUPPORT_PROVENANCE_KEYS
                }
            )
        if cameras is not None:
            data["camera_order"] = tuple(cameras)
        known = {item for item in cls.__dataclass_fields__}
        data = {key: value for key, value in data.items() if key in known}
        return cls(**data)


class N01V12Model(nn.Module):
    """One complete arm: content path, measurement support and readout bridge.

    The external encoder contract (``init_state`` / ``step`` / ``readout`` /
    ``reset`` / ``detach_state`` / ``state_bytes``) is the one already in force, so
    this class is a factory and a thin delegation rather than a parallel interface.
    Support is analytic, so ``raw_n01`` and ``raw_base`` report the same parameter
    count; the difference between them is the content that reaches the fusion.
    """

    def __init__(self, config: N01V12Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = DbsrFeatureAdapter(config.adapter_config())
        support_type = (
            WindowGLSSupport if config.support.implementation == "gls_window" else SpatialSupport
        )
        self.support = support_type(config.support) if config.spec.support_enabled else None
        if self.support is not None:
            self.encoder.set_support(self.support)
        if config.motion is not None:
            self.encoder.content_frame_chunk = 1
            self.encoder.checkpoint_content = True
            self.encoder.set_motion_provider(
                DomainWindowMotion(config.motion, config.spec.input_domain)
            )
        self.bridge = self.encoder.build_postprocessor(self.config.cameras)

    # -- external contract -------------------------------------------------------

    def init_state(self, batch_size: int, camera_spec: Any, device: Any = "cpu") -> FullState:
        if self.config.observation is not None and any(self.config.observation.output_shape):
            camera_spec = {**camera_spec, "shape": self.config.observation.output_shape}
        return self.encoder.init_state(batch_size, camera_spec, device)

    def step(
        self,
        frame: MeasurementFrame,
        embodiment: Any = None,
        state: FullState | None = None,
        budget: Any = None,
        offsets_xy: Any = None,
    ) -> EncoderOutput:
        if self.config.observation is not None:
            frame = prepare_frame(frame, self.config.observation, self.config.spec.input_domain)
        return self.encoder.step(
            frame, embodiment=embodiment, state=state, budget=budget, offsets_xy=offsets_xy
        )

    def readout(self, output: EncoderOutput, token_budget: int | None = None) -> VisualTokens:
        return self.encoder.readout(output, token_budget=token_budget)

    def reset(self, state: FullState, batch_mask: Any = None) -> FullState:
        return self.encoder.reset(state, batch_mask=batch_mask)

    def detach_state(self, state: FullState) -> FullState:
        return self.encoder.detach_state(state)

    def state_bytes(self, state: FullState) -> int:
        return self.encoder.state_bytes(state)

    def delete_sources(self, state, camera, source_ids, selected=None):
        from owac.frameworks.n01_full.source_replay import delete_sources

        return delete_sources(self, state, camera, source_ids, selected)

    def fixed_decision_readout(self, state, camera, source_ids):
        from owac.frameworks.n01_full.source_replay import fixed_decision_readout

        return fixed_decision_readout(self, state, camera, source_ids)

    # -- reports -----------------------------------------------------------------

    def parameter_report(self) -> dict[str, Any]:
        encoder = self.encoder.parameter_report()
        bridge = sum(parameter.numel() for parameter in self.bridge.parameters())
        encoder_total = int(encoder["total"])
        return {
            "encoder": encoder_total,
            "bridge": int(bridge),
            "total_trainable": encoder_total + int(bridge),
            "support_parameters": 0,
            "note": "support is analytic: N01 and Base arms share one parameter count",
        }

    def arm_report(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "stem": self.encoder.stem_report(),
            "parameters": self.parameter_report(),
            "support_enabled": bool(self.support is not None),
        }


def arm_matrix(
    *,
    platforms: tuple[str, ...] = PLATFORMS,
    methods: tuple[str, ...] = ARM_METHODS,
    seed: int = DEFAULT_SEED,
    window_frames: int = 32,
    complete: bool = True,
) -> list[N01V12Config]:
    """The eight main runs: two platforms x four arms x one training seed.

    The single seed is the current instruction.  It is *not* evidence about
    training-seed variance, and the report must say so rather than implying a
    stability claim that one seed cannot support.
    """

    return [
        N01V12Config(
            arm=method,
            platform=platform,
            seed=seed,
            window_frames=window_frames,
            motion=MotionConfig() if complete else None,
            observation=ObservationConfig(
                native_shape=(256, 256) if platform == "libero" else (240, 320),
                shot=4e-4 if method.startswith("raw") and platform == "libero" else 0.0,
                read_variance=(1e-5 + (1 / 255) ** 2 / 12)
                if method.startswith("raw") and platform == "libero"
                else 1e-4,
                noise_provenance="author_libero_shot_read_plus_quantisation"
                if method.startswith("raw") and platform == "libero"
                else "declared_same_domain_proxy_unfitted",
            )
            if complete
            else None,
            support=SupportConfig(
                implementation="gls_window",
                large_residual_threshold=3.0,
                prior_mean=1.0,
                obs_var=0.25,
            )
            if complete
            else SupportConfig(),
            training=TrainingPlan() if complete else None,
            supported_logic_version="n01-v1.2-p3-complete-20261017" if complete else "n01-v1.2-p3",
        )
        for platform in platforms
        for method in methods
    ]


def build_model(config: N01V12Config, *, device: Any = "cpu") -> N01V12Model:
    """Factory used by training, evaluation and the runtime service.

    Every entry point goes through here so that no caller can silently fall back
    to the prototype trainer's old factory.
    """

    model = N01V12Model(config)
    return model.to(torch.device(device)) if device is not None else model

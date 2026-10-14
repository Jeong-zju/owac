"""N01 v1.0 P2: full-size DBSR content adapter behind the external encoder contract.

``docs/prompts/n01_v1_0/00_SHARED_CONTRACT.md`` sections 2/3 and
``docs/prompts/n01_v1_0/EARLY_POLICY_PREFLIGHT.md`` (``mode=raw_adapter``) fix
this module's job:

* reuse the repository's full DBSR content branch (64 channels, 9 residual
  blocks, 512-wide features) and :class:`MaskedDbsrFusion`;
* replace the *input stem* with an explicit three-channel convolution for the
  author ``rawvla_rgb3`` pseudo-RAW, while the packed-CFA four-channel entry
  keeps the untouched upstream stem and its canonical RGGB phase contract;
* keep one independent measurement buffer per camera, process exactly one
  arrival event per :meth:`DbsrFeatureAdapter.step`, and let a repeated frame
  advance the clock without adding evidence;
* pool the 512-wide spatial field deterministically with the validity mask to a
  16x16 grid per camera (``pool_grid ** 2 == tokens_per_camera == 256``), then
  let the caller project it with the shared
  ``VisualTokenPostprocessor(in_dim=512, width=2048, tokens_per_camera=256)``
  built by :meth:`DbsrFeatureAdapter.build_postprocessor`.

Honest boundaries:

* P3 support/retraction logic is hard-off (:data:`P3_SUPPORT_ENABLED`).  The
  support graph is never applied and ``diagnostics["p3_support_applied"]`` is
  always ``False``; P4 reuses this adapter instead of building a second numeric
  conversion.
* P2 motion is wired through :meth:`DbsrFeatureAdapter.set_motion_provider` or an
  explicit ``offsets_xy`` (``[B,T,2,H,W]`` packed-cell pull offsets for the
  fusion).  The provider receives the exact ``window`` the step is fusing and
  returns that many pairs; the adapter validates ``T`` and the grid and never
  rewrites a unit.  :class:`owac.frameworks.n01_full.motion_builders.WindowMotionProvider`
  is the first-version provider over the motion-core estimator; with no provider
  the window is fused with identity offsets.  The per-camera motion provenance is
  kept in :class:`CameraCorrespondence` (``FullState.correspondence``) for P4.
* The only projection from the 512-wide pooled field to the language width is
  the postprocessor's counted ``in_proj`` (512 -> 2048).  This module allocates
  no additional 2048 -> 2048 (or any other) readout projection.
* No RGB/sRGB/XYZ intermediate is produced anywhere on this path.

The package ``__init__`` never imports this module, so ``import
owac.frameworks.n01_full`` stays torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from owac.backends.rawvla_features import (
    CONTROLLED_CONFIG_NAME,
    VisualPrefixConfig,
    VisualTokenPostprocessor,
    parameter_report,
)
from owac.baselines._vendor.dbsr import blocks as dbsr_blocks
from owac.baselines.burst_features import (
    BurstFeatureConfig,
    MaskedDbsrFusion,
    RawBurstBackbone,
    align_features,
)
from owac.core.encoder_contract import ComputeBudget, EncoderOutput, MeasurementFrame, VisualTokens
from owac.frameworks.n01 import state as measurement_state
from owac.frameworks.n01.config import N01Config

__all__ = [
    "CFA_INPUT_MODES",
    "DbsrAdapterConfig",
    "DbsrFeatureAdapter",
    "FullState",
    "CameraCorrespondence",
    "CameraTokens",
    "P3_SUPPORT_ENABLED",
    "POOL_GRID",
    "THREE_CHANNEL_INPUT_MODES",
    "pooled_token_positions",
]

CFA_INPUT_MODES = ("bayer_cfa", "synthetic_cfa")
THREE_CHANNEL_INPUT_MODES = ("rawvla_rgb3",)

POOL_GRID = 16
"""Deployed per-camera token grid: 16 x 16 == contract ``readout_tokens`` 256."""

P3_SUPPORT_ENABLED = False
"""Hard-off in P2: no support graph, hypothesis merge/split or retraction is applied here.

The flag is a module constant, not a runtime option, so a run cannot silently
claim P3 support behaviour: :meth:`DbsrFeatureAdapter.step` always reports
``p3_support_applied == False``.
"""

_CFA_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class DbsrAdapterConfig:
    """First-version adapter configuration (full-size DBSR defaults)."""

    input_mode: str = "rawvla_rgb3"
    feature_width: int = 512
    stem_width: int = 64
    residual_blocks: int = 9
    fusion: str = "dbsr_reference"
    projection_width: int = 32
    offset_width: int = 64
    max_history: int = 8
    pool_grid: int = POOL_GRID
    readout_width: int = 2048
    tokens_per_camera: int = 256
    cfa_pattern: str = "RGGB"

    def __post_init__(self) -> None:
        if self.input_mode not in CFA_INPUT_MODES + THREE_CHANNEL_INPUT_MODES:
            raise ValueError(
                f"unknown input_mode {self.input_mode!r}; expected one of "
                f"{CFA_INPUT_MODES + THREE_CHANNEL_INPUT_MODES}"
            )
        if self.pool_grid < 1:
            raise ValueError("pool_grid must be positive")
        if self.pool_grid * self.pool_grid != self.tokens_per_camera:
            raise ValueError(
                f"tokens_per_camera {self.tokens_per_camera} != pool_grid^2 "
                f"{self.pool_grid * self.pool_grid}"
            )
        if self.max_history < 1 or self.max_history > 8:
            raise ValueError("the exact unique-frame buffer keeps at most 8 frames")
        for name, value in (
            ("feature_width", self.feature_width),
            ("stem_width", self.stem_width),
            ("residual_blocks", self.residual_blocks),
            ("projection_width", self.projection_width),
            ("offset_width", self.offset_width),
            ("readout_width", self.readout_width),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.is_cfa and self.cfa_pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            raise ValueError(f"unknown CFA pattern {self.cfa_pattern!r}")

    @property
    def is_cfa(self) -> bool:
        return self.input_mode in CFA_INPUT_MODES

    @property
    def input_channels(self) -> int:
        return 4 if self.is_cfa else 3

    @property
    def planes(self) -> int:
        return 4 if self.is_cfa else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "input_channels": self.input_channels,
            "feature_width": self.feature_width,
            "stem_width": self.stem_width,
            "residual_blocks": self.residual_blocks,
            "fusion": self.fusion,
            "projection_width": self.projection_width,
            "offset_width": self.offset_width,
            "max_history": self.max_history,
            "pool_grid": self.pool_grid,
            "tokens_per_camera": self.tokens_per_camera,
            "readout_width": self.readout_width,
            "cfa_pattern": self.cfa_pattern if self.is_cfa else None,
        }


@dataclass
class CameraTokens:
    """Readout-ready pooled tokens of one camera (512-wide, before the bridge)."""

    values: torch.Tensor  # [B, tokens_per_camera, feature_width]
    mask: torch.Tensor  # [B, tokens_per_camera] bool, False = invalid cell
    positions: torch.Tensor  # [B, tokens_per_camera, 2] in [-1, 1]
    window_frames: int  # unique frames fused into this encoding
    newest_frame_ids: torch.Tensor  # [B] int64 identity of the newest source


@dataclass
class CameraCorrespondence:
    """Motion provenance of one camera's latest fused window (P2 hand-off to P4).

    ``offsets_xy`` is the field that was actually handed to the fusion in the
    fusion's own packed-cell units; the provider owns the sensor<->packed unit
    conversion and the four-phase offsets.  ``valid`` marks which window pairs
    produced an accepted map, and ``risk_kind`` distinguishes an uncalibrated
    proxy from an independently calibrated table.
    """

    mapping_version: str
    decision_version: str
    risk_kind: str
    window_frames: int
    valid: torch.Tensor  # [B, T] bool
    source_frame_ids: tuple[Any, ...]
    offsets_xy: torch.Tensor  # [B, T, 2, H, W] packed-cell pull offsets
    risk: torch.Tensor | None = None  # [B, T, H, W]
    visibility: torch.Tensor | None = None  # [B, T, H, W]

    def describe(self) -> dict[str, Any]:
        return {
            "mapping_version": self.mapping_version,
            "decision_version": self.decision_version,
            "risk_kind": self.risk_kind,
            "window_frames": int(self.window_frames),
            "valid_pairs_per_member": [int(value) for value in self.valid.sum(-1).tolist()],
            "source_frame_ids": [
                None if item is None else int(item) for item in self.source_frame_ids
            ],
            "offsets_shape": [int(size) for size in self.offsets_xy.shape],
        }


@dataclass
class FullState:
    """One independent measurement buffer, token cache and motion record per camera."""

    camera_order: tuple[str, ...]
    batch_size: int
    device: torch.device
    buffer_config: N01Config
    cameras: dict[str, Any] = field(default_factory=dict)
    episodes: dict[str, list[Any]] = field(default_factory=dict)
    tokens: dict[str, CameraTokens] = field(default_factory=dict)
    correspondence: dict[str, CameraCorrespondence] = field(default_factory=dict)
    arrivals: dict[str, torch.Tensor] = field(default_factory=dict)
    unique_arrivals: dict[str, torch.Tensor] = field(default_factory=dict)
    step_index: int = 0

    @property
    def camera_ids(self) -> list[str]:
        return list(self.camera_order)

    def has_camera(self, camera_id: str) -> bool:
        return camera_id in self.camera_order

    def tokens_for(self, camera_id: str) -> CameraTokens | None:
        return self.tokens.get(camera_id)

    def correspondence_for(self, camera_id: str) -> CameraCorrespondence | None:
        return self.correspondence.get(camera_id)


def pooled_token_positions(grid: int, *, device: Any = None, dtype: Any = torch.float32) -> Any:
    """Cell-centre positions in [-1, 1], row-major (token ``k = row * grid + col``)."""
    rows = torch.arange(grid, device=device, dtype=dtype)
    cols = torch.arange(grid, device=device, dtype=dtype)
    x = -1.0 + (2.0 * cols + 1.0) / grid
    y = -1.0 + (2.0 * rows + 1.0) / grid
    mesh = torch.meshgrid(x, y, indexing="xy")
    return torch.stack(mesh, dim=-1).reshape(grid * grid, 2)


def _require_canonical_rggb(sample_xy: Any, device: Any) -> None:
    """The packed CFA entry keeps the old explicit canonical RGGB phase contract."""
    if sample_xy is None:
        raise ValueError("bayer_cfa input requires explicit canonical RGGB sample_xy phases")
    phases = torch.as_tensor(sample_xy, device=device)
    expected = torch.tensor(_CFA_PHASES, device=device)
    if phases.shape != expected.shape or not torch.equal(phases, expected):
        raise ValueError("explicit canonical RGGB sample_xy phases required")


def _parse_camera_spec(camera_spec: Any) -> tuple[list[str], tuple[int, int] | None, int | None]:
    """Accept a camera id string, a sequence of ids, or a mapping with a declared shape."""
    declared_channels: int | None = None
    if isinstance(camera_spec, str):
        camera_ids, shape = [camera_spec], None
    elif isinstance(camera_spec, dict):
        camera_ids = list(camera_spec.get("camera_ids") or [])
        if not camera_ids and camera_spec.get("camera_id"):
            camera_ids = [str(camera_spec["camera_id"])]
        shape = camera_spec.get("shape")
        if shape is not None:
            shape = (int(shape[0]), int(shape[1]))
        if camera_spec.get("channels") is not None:
            declared_channels = int(camera_spec["channels"])
    else:
        camera_ids, shape = list(camera_spec), None
    if not camera_ids:
        raise ValueError("camera_spec must name at least one camera")
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError(f"duplicate camera ids in camera_spec: {camera_ids}")
    return camera_ids, shape, declared_channels


def _episode_values(episode_ids: Any, batch_size: int) -> list[Any]:
    """Plain Python episode values, whether the caller passes a tensor or a sequence."""
    if torch.is_tensor(episode_ids):
        values = episode_ids.detach().cpu().reshape(-1).tolist()
    else:
        values = list(episode_ids)
    if len(values) != batch_size:
        raise ValueError(f"episode_ids must list {batch_size} entries, got {len(values)}")
    return values


def _batch_selector(batch_mask: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    """Normalize ``None`` / bool mask / index list into a ``[B]`` bool selector."""
    if batch_mask is None:
        return torch.ones(batch_size, dtype=torch.bool, device=device)
    if (
        isinstance(batch_mask, (list, tuple))
        and batch_mask
        and all(isinstance(item, bool) for item in batch_mask)
    ):
        selector = torch.tensor(list(batch_mask), dtype=torch.bool, device=device)
    else:
        tensor = torch.as_tensor(batch_mask, device=device)
        if tensor.dtype == torch.bool:
            selector = tensor.reshape(-1).to(torch.bool)
        else:
            selector = torch.zeros(batch_size, dtype=torch.bool, device=device)
            selector[tensor.reshape(-1).long()] = True
    if selector.numel() == 1:
        selector = selector.expand(batch_size)
    if selector.numel() != batch_size:
        raise ValueError(f"batch_mask must select within the batch of {batch_size}")
    return selector


class DbsrFeatureAdapter(nn.Module):
    """Full DBSR content path as an external-contract encoder.

    ``backbone`` is the repository's full-size DBSR branch and ``fusion`` the
    mask-aware :class:`MaskedDbsrFusion`; neither upstream module is modified.
    Only the three-channel ``rawvla_rgb3`` stem is created here (the upstream
    four-channel ``init_layer`` is replaced by a three-channel convolution);
    the packed-CFA path leaves the upstream stem untouched.
    """

    p3_support_enabled = P3_SUPPORT_ENABLED

    def __init__(self, config: DbsrAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or DbsrAdapterConfig()
        burst_config = BurstFeatureConfig(
            fusion=self.config.fusion,
            stem_width=self.config.stem_width,
            residual_blocks=self.config.residual_blocks,
            feature_width=self.config.feature_width,
            projection_width=self.config.projection_width,
            offset_width=self.config.offset_width,
            max_history=self.config.max_history,
        )
        self.burst_config = burst_config
        self.backbone = RawBurstBackbone(burst_config)
        self.fusion = MaskedDbsrFusion(burst_config)
        self.stem_replaced = False
        if not self.config.is_cfa:
            upstream = self.backbone.upstream
            upstream.init_layer = dbsr_blocks.conv_block(
                self.config.input_channels,
                self.config.stem_width,
                3,
                stride=1,
                padding=1,
                batch_norm=False,
                activation="relu",
            )
            self.stem_replaced = True
        self.motion_provider: Callable[..., Any] | None = None

    def set_motion_provider(self, provider: Callable[..., Any] | None) -> None:
        """Attach the P2 window-motion provider (``frame, state, camera_id, window``).

        The provider receives the camera's post-push :class:`N01State` and the exact
        ``window`` the step is fusing, so it cannot recompute or extend the window.
        It returns a tensor ``[B, T, 2, H, W]`` or a rich object exposing
        ``offsets_xy`` (plus optional ``valid``/``risk``/``risk_kind``/
        ``mapping_version``/``decision_version``/``source_frame_ids``); the sensor
        coordinate and packed/sensor unit conversion belongs to the provider.
        """
        self.motion_provider = provider

    # -- derived contracts ---------------------------------------------------------

    @property
    def buffer_config(self) -> N01Config:
        """Unique-measurement buffer only: no support graph, no auxiliary RAW head."""
        return N01Config.tiny(
            input_mode="bayer_cfa" if self.config.is_cfa else "rawvla_rgb3",
            channels=self.config.input_channels,
            planes=self.config.planes,
            cfa_pattern=self.config.cfa_pattern,
            spatial_radii=(0,),
            time_depths=(self.config.max_history,),
            max_history=self.config.max_history,
            require_unique_frames=True,
            aux_measurement_prediction=False,
        )

    def stem_report(self) -> dict[str, Any]:
        return {
            "input_channels": self.config.input_channels,
            "stem_replaced": bool(self.stem_replaced),
            "upstream_default_channels": 4,
            "cfa_entry_channels": 4,
        }

    def parameter_report(self) -> dict[str, Any]:
        """Exact counts; the bridge projection is counted separately in the postprocessor."""
        return {
            "backbone": parameter_report(self.backbone),
            "fusion": parameter_report(self.fusion),
            "total": int(sum(parameter.numel() for parameter in self.parameters())),
        }

    def build_postprocessor(
        self,
        camera_order: Sequence[str],
        *,
        name: str = CONTROLLED_CONFIG_NAME,
        spatial_mode: str = "auto",
    ) -> VisualTokenPostprocessor:
        """The shared readout projection: 512 -> 2048, 256 tokens per camera.

        ``in_dim`` is passed explicitly, so the projection can never silently
        fall back to the ``from_readout_config`` default (2048 -> 2048).
        """
        config = VisualPrefixConfig(
            width=self.config.readout_width,
            in_dim=self.config.feature_width,
            tokens_per_camera=self.config.tokens_per_camera,
            camera_order=tuple(camera_order),
            spatial_mode=spatial_mode,
            name=name,
        )
        return VisualTokenPostprocessor(config)

    # -- external contract ---------------------------------------------------------

    def init_state(self, batch_size: int, camera_spec: Any, device: Any = "cpu") -> FullState:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        camera_ids, shape, declared_channels = _parse_camera_spec(camera_spec)
        if declared_channels is not None and declared_channels != self.config.input_channels:
            raise ValueError(
                f"camera_spec declares {declared_channels} channels but this adapter takes "
                f"{self.config.input_channels} ({self.config.input_mode})"
            )
        device = torch.device(device)
        cameras: dict[str, Any] = {}
        if shape is not None:
            for camera in camera_ids:
                cameras[camera] = measurement_state.init_state(
                    self.buffer_config,
                    int(batch_size),
                    {"camera_ids": [camera], "shape": shape},
                    device,
                )
        zeros = {
            camera: torch.zeros(int(batch_size), dtype=torch.int64, device=device)
            for camera in camera_ids
        }
        return FullState(
            camera_order=tuple(camera_ids),
            batch_size=int(batch_size),
            device=device,
            buffer_config=self.buffer_config,
            cameras=cameras,
            episodes={camera: [None] * int(batch_size) for camera in camera_ids},
            tokens={},
            arrivals=dict(zeros),
            unique_arrivals=dict(zeros),
        )

    def step(
        self,
        frame: MeasurementFrame,
        embodiment: Any = None,
        state: FullState | None = None,
        budget: ComputeBudget | None = None,
        offsets_xy: Any = None,
    ) -> EncoderOutput:
        """Observe one arrival event of one camera; a repeat never adds evidence."""
        del embodiment  # accepted for the shared contract; unused on the content path
        if state is None:
            raise ValueError("DbsrFeatureAdapter.step requires a state from init_state")
        if not isinstance(state, FullState):
            raise TypeError(f"state must be a FullState, got {type(state).__name__}")
        values = torch.as_tensor(frame.values)
        if not torch.is_tensor(values) or values.ndim != 4:
            raise ValueError("MeasurementFrame.values must be a [B,C,H,W] tensor")
        batch, channels, height, width = (int(size) for size in values.shape)
        if batch != state.batch_size:
            raise ValueError(f"frame batch {batch} does not match state batch {state.batch_size}")
        if channels != self.config.input_channels:
            raise ValueError(
                f"frame has {channels} channels but {self.config.input_mode} expects "
                f"{self.config.input_channels}"
            )
        if self.config.is_cfa:
            _require_canonical_rggb(frame.sample_xy, values.device)
        camera = self._resolve_camera(frame, state)
        budget = budget or ComputeBudget(max_updates=1, max_history=self.config.max_history)
        if budget.max_history < 1:
            raise ValueError("budget.max_history must be positive")
        if budget.max_history > self.config.max_history:
            raise ValueError("budget.max_history exceeds the configured unique-frame capacity")

        # New-episode first arrival: reset that member *before* the frame is pushed.
        episode_ids = frame.episode_ids
        if episode_ids is not None:
            reset_mask = self._episode_reset_mask(state, camera, episode_ids)
            if bool(reset_mask.any()):
                state = self.reset(state, reset_mask)
        current = state.cameras.get(camera)
        unique = (
            torch.ones(batch, dtype=torch.bool, device=state.device)
            if current is None
            else self._unique_flags(current, frame, batch, state.device)
        )
        if current is None:
            current = measurement_state.init_state(
                self.buffer_config,
                batch,
                {"camera_ids": [camera], "shape": (height, width)},
                state.device,
            )
        updated = measurement_state.push(current, frame)
        window = min(int(updated.filled.max()), int(budget.max_history))
        tokens = dict(state.tokens)
        correspondence = dict(state.correspondence)
        pooled_mask = None
        weights = None
        source_frame_ids = None
        motion_report: dict[str, Any] = {"status": "identity_pending_p2"}
        if window > 0:
            window_values, cell_mask = self._window(updated, window)
            motion = offsets_xy
            motion_source = "explicit_offsets" if offsets_xy is not None else "identity_pending_p2"
            if motion is None and self.motion_provider is not None:
                motion = self.motion_provider(
                    frame=frame, state=updated, camera_id=camera, window=window
                )
                motion_source = "motion_provider"
            resolved_offsets, motion_report = self._resolve_motion(motion, updated, window, batch)
            motion_report["status"] = motion_source
            fused, available, weights = self.encode_window(
                window_values, cell_mask, resolved_offsets
            )
            pooled, pooled_mask, positions = self._masked_pool(fused, available)
            tokens[camera] = CameraTokens(
                values=pooled,
                mask=pooled_mask,
                positions=positions,
                window_frames=window,
                newest_frame_ids=updated.frame_ids[0].detach(),
            )
            default_ids = tuple(int(value) for value in updated.frame_ids[:window, 0].tolist())
            correspondence[camera] = CameraCorrespondence(
                mapping_version=motion_report["mapping_version"],
                decision_version=motion_report["decision_version"],
                risk_kind=motion_report["risk_kind"],
                window_frames=window,
                valid=motion_report["valid"],
                source_frame_ids=motion_report.get("source_frame_ids") or default_ids,
                offsets_xy=(
                    torch.zeros(
                        batch,
                        window,
                        2,
                        *window_values.shape[-2:],
                        dtype=window_values.dtype,
                        device=window_values.device,
                    )
                    if resolved_offsets is None
                    else resolved_offsets.detach()
                ),
                risk=motion_report.get("risk"),
                visibility=motion_report.get("visibility"),
            )
            source_frame_ids = updated.frame_ids[:window].detach()
        else:
            # Nothing unique has ever arrived for this camera: no token block.
            tokens.pop(camera, None)
            correspondence.pop(camera, None)
        new_state = replace(
            state,
            cameras={**state.cameras, camera: updated},
            tokens=tokens,
            correspondence=correspondence,
            arrivals={
                **state.arrivals,
                camera: state.arrivals.get(camera, torch.zeros_like(unique, dtype=torch.int64)) + 1,
            },
            unique_arrivals={
                **state.unique_arrivals,
                camera: state.unique_arrivals.get(
                    camera, torch.zeros_like(unique, dtype=torch.int64)
                )
                + unique.to(torch.int64),
            },
            episodes=(
                state.episodes
                if episode_ids is None
                else {**state.episodes, camera: list(_episode_values(episode_ids, batch))}
            ),
            step_index=state.step_index + 1,
        )
        validity = {cid: entry.mask for cid, entry in new_state.tokens.items()}
        diagnostics: dict[str, Any] = {
            "camera_id": camera,
            "arrival_index": int(new_state.step_index),
            "window_frames": int(window),
            "unique_frames_this_step": [bool(flag) for flag in unique.tolist()],
            "repeated_this_step": [not bool(flag) for flag in unique.tolist()],
            "unique_frames_lifetime": [
                int(value) for value in new_state.unique_arrivals[camera].tolist()
            ],
            "arrivals_lifetime": [int(value) for value in new_state.arrivals[camera].tolist()],
            "source_frame_ids": source_frame_ids,
            "fusion_weights": None if weights is None else weights.detach(),
            "pooled_token_mask": pooled_mask,
            "p3_support_applied": False,
            "p3_support_status": "disabled_pending_p4",
            "p2_motion_status": motion_report["status"],
            "p2_motion": {
                key: value
                for key, value in motion_report.items()
                if key in ("mapping_version", "decision_version", "risk_kind", "valid_pairs")
            },
            "embodiment_used": False,
            "aux_raw_loss": False,
            "variance_used_as_fusion_weight": False,
        }
        return EncoderOutput(
            representation={cid: entry.values for cid, entry in new_state.tokens.items()},
            state=new_state,
            validity=validity,
            diagnostics=diagnostics,
            cost={
                "state_bytes": self.state_bytes(new_state),
                "arrivals": int(new_state.step_index),
            },
        )

    def readout(self, output: EncoderOutput, token_budget: int | None = None) -> VisualTokens:
        """Project the cached per-camera tokens for the bridge; never observes."""
        if not isinstance(output, EncoderOutput):
            raise TypeError(f"readout expects an EncoderOutput, got {type(output).__name__}")
        state = output.state
        if not isinstance(state, FullState):
            raise TypeError("EncoderOutput.state must be the FullState returned by step")
        if token_budget is not None and int(token_budget) != self.config.tokens_per_camera:
            raise ValueError(
                f"token_budget {token_budget} != fixed pool grid "
                f"{self.config.tokens_per_camera}; the grid is fixed by pooling"
            )
        camera_ids = [camera for camera in state.camera_order if camera in state.tokens]
        if not camera_ids:
            raise RuntimeError("no camera holds a unique measurement yet; nothing to read out")
        values = torch.cat([state.tokens[camera].values for camera in camera_ids], dim=1)
        mask = torch.cat([state.tokens[camera].mask for camera in camera_ids], dim=1)
        positions = torch.cat([state.tokens[camera].positions for camera in camera_ids], dim=1)
        return VisualTokens(
            values=values, mask=mask, camera_ids=list(camera_ids), positions=positions
        )

    def reset(self, state: FullState, batch_mask: Any = None) -> FullState:
        """Reset only the selected batch members; other episodes keep their buffers."""
        if not isinstance(state, FullState):
            raise TypeError(f"reset expects a FullState, got {type(state).__name__}")
        selected = _batch_selector(batch_mask, state.batch_size, state.device)
        cameras = {
            camera: measurement_state.reset(camera_state, selected)
            for camera, camera_state in state.cameras.items()
        }
        tokens: dict[str, CameraTokens] = {}
        for camera, entry in state.tokens.items():
            keep = ~selected
            if not bool(keep.any()):
                continue
            tokens[camera] = CameraTokens(
                values=entry.values.masked_fill(~keep[:, None, None], 0),
                mask=entry.mask & keep[:, None],
                positions=entry.positions.masked_fill(~keep[:, None, None], 0),
                window_frames=entry.window_frames,
                newest_frame_ids=torch.where(
                    keep, entry.newest_frame_ids, torch.full_like(entry.newest_frame_ids, -1)
                ),
            )
        episodes = {
            camera: [None if bool(selected[i]) else value for i, value in enumerate(values)]
            for camera, values in state.episodes.items()
        }
        correspondence: dict[str, CameraCorrespondence] = {}
        for camera, entry in state.correspondence.items():
            keep = ~selected
            if not bool(keep.any()):
                continue
            correspondence[camera] = CameraCorrespondence(
                mapping_version=entry.mapping_version,
                decision_version=entry.decision_version,
                risk_kind=entry.risk_kind,
                window_frames=entry.window_frames,
                valid=entry.valid & keep[:, None],
                source_frame_ids=entry.source_frame_ids,
                offsets_xy=entry.offsets_xy.masked_fill(~keep[:, None, None, None, None], 0),
                risk=(
                    None
                    if entry.risk is None
                    else entry.risk.masked_fill(~keep[:, None, None, None], 0)
                ),
                visibility=(
                    None
                    if entry.visibility is None
                    else entry.visibility.masked_fill(~keep[:, None, None, None], 0)
                ),
            )
        return replace(
            state,
            cameras=cameras,
            tokens=tokens,
            correspondence=correspondence,
            episodes=episodes,
            arrivals={
                camera: torch.where(selected, torch.zeros_like(values), values)
                for camera, values in state.arrivals.items()
            },
            unique_arrivals={
                camera: torch.where(selected, torch.zeros_like(values), values)
                for camera, values in state.unique_arrivals.items()
            },
        )

    def detach_state(self, state: FullState) -> FullState:
        """Detach every tensor at a TBPTT boundary without changing any value."""
        if not isinstance(state, FullState):
            raise TypeError(f"detach_state expects a FullState, got {type(state).__name__}")
        return replace(
            state,
            cameras={
                camera: measurement_state.detach(camera_state)
                for camera, camera_state in state.cameras.items()
            },
            tokens={
                camera: CameraTokens(
                    values=entry.values.detach(),
                    mask=entry.mask.detach(),
                    positions=entry.positions.detach(),
                    window_frames=entry.window_frames,
                    newest_frame_ids=entry.newest_frame_ids.detach(),
                )
                for camera, entry in state.tokens.items()
            },
            correspondence={
                camera: CameraCorrespondence(
                    mapping_version=entry.mapping_version,
                    decision_version=entry.decision_version,
                    risk_kind=entry.risk_kind,
                    window_frames=entry.window_frames,
                    valid=entry.valid.detach(),
                    source_frame_ids=entry.source_frame_ids,
                    offsets_xy=entry.offsets_xy.detach(),
                    risk=None if entry.risk is None else entry.risk.detach(),
                    visibility=None if entry.visibility is None else entry.visibility.detach(),
                )
                for camera, entry in state.correspondence.items()
            },
            arrivals={camera: value.detach() for camera, value in state.arrivals.items()},
            unique_arrivals={
                camera: value.detach() for camera, value in state.unique_arrivals.items()
            },
        )

    def state_bytes(self, state: FullState) -> int:
        total = 0
        for camera_state in state.cameras.values():
            total += measurement_state.state_bytes(camera_state)
        for entry in state.tokens.values():
            for tensor in (entry.values, entry.mask, entry.positions, entry.newest_frame_ids):
                total += int(tensor.numel()) * int(tensor.element_size())
        for entry in state.correspondence.values():
            for tensor in (entry.valid, entry.offsets_xy, entry.risk, entry.visibility):
                if tensor is not None:
                    total += int(tensor.numel()) * int(tensor.element_size())
        for table in (state.arrivals, state.unique_arrivals):
            for tensor in table.values():
                total += int(tensor.numel()) * int(tensor.element_size())
        return total

    # -- internals -----------------------------------------------------------------

    def encode_window(
        self, values: torch.Tensor, cell_mask: torch.Tensor, offsets_xy: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DBSR-encode a ``[B,T,C,H,W]`` window (newest first) and fuse it."""
        batch, length, channels, height, width = values.shape
        encoded = self.backbone(values.reshape(batch * length, channels, height, width))
        encoded = encoded.reshape(batch, length, self.config.feature_width, height, width)
        encoded = encoded.masked_fill(~cell_mask, 0)
        aligned, valid, offsets = align_features(encoded, cell_mask, offsets_xy)
        fused, weights = self.fusion(aligned, valid, offsets)
        available = valid.any(1)
        return fused.masked_fill(~available, 0), available, weights

    def _resolve_motion(
        self, motion: Any, camera_state: Any, window: int, batch: int
    ) -> tuple[Any, dict[str, Any]]:
        """Normalize the provider/kwarg result; the adapter never rewrites units.

        The provider owns the sensor<->packed conversion and the four-phase
        offsets; this method only checks that the packed-cell field really covers
        the window the step is fusing (``T`` is never recomputed here).
        """
        device = camera_state.values.device
        if motion is None:
            return None, {
                "mapping_version": "identity",
                "decision_version": "identity",
                "risk_kind": "unavailable",
                "valid": torch.ones(batch, window, dtype=torch.bool, device=device),
                "risk": None,
                "visibility": None,
                "source_frame_ids": (),
                "valid_pairs": [window] * batch,
            }
        offsets = getattr(motion, "offsets_xy", motion)
        if not torch.is_tensor(offsets) or offsets.ndim != 5 or int(offsets.shape[2]) != 2:
            raise ValueError(
                "motion offsets must be a [B, T, 2, H, W] tensor or expose .offsets_xy"
            )
        if int(offsets.shape[0]) != batch or int(offsets.shape[1]) != window:
            raise ValueError(
                f"motion offsets {tuple(offsets.shape)} must cover the fused window "
                f"[B={batch}, T={window}, 2, H, W]; a motion provider cannot change the window"
            )
        expected_grid = tuple(int(size) for size in camera_state.shape)
        if tuple(int(size) for size in offsets.shape[-2:]) != expected_grid:
            raise ValueError(
                f"motion offsets grid {tuple(offsets.shape[-2:])} != buffer grid {expected_grid}"
            )
        valid = getattr(motion, "valid", None)
        if valid is None:
            valid = torch.ones(batch, window, dtype=torch.bool, device=device)
        else:
            valid = torch.as_tensor(valid, device=device, dtype=torch.bool).detach()
            if tuple(valid.shape) != (batch, window):
                raise ValueError(
                    f"motion valid mask {tuple(valid.shape)} does not match {(batch, window)}"
                )
        risk = getattr(motion, "risk", None)
        visibility = getattr(motion, "visibility", None)
        return offsets.to(device=device, dtype=torch.float32), {
            "mapping_version": str(getattr(motion, "mapping_version", "unknown")),
            "decision_version": str(
                getattr(motion, "decision_version", None) or getattr(motion, "version", "unknown")
            ),
            "risk_kind": str(getattr(motion, "risk_kind", "unavailable")),
            "valid": valid,
            "risk": (
                None
                if risk is None
                else torch.as_tensor(risk, device=device, dtype=torch.float32).detach()
            ),
            "visibility": (
                None
                if visibility is None
                else torch.as_tensor(visibility, device=device, dtype=torch.float32).detach()
            ),
            "source_frame_ids": tuple(getattr(motion, "source_frame_ids", ())),
            "valid_pairs": [int(value) for value in valid.sum(-1).tolist()],
        }

    def _window(self, camera_state: Any, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        values = camera_state.values[:count].transpose(0, 1)
        valid = camera_state.valid[:count].transpose(0, 1)
        observed = camera_state.observed[:count].T[:, :, None, None, None]
        valid = valid & observed & torch.isfinite(values)
        # Invalid samples contribute zero to the convolution, exactly as in the accepted
        # baseline: a bad pixel's value never enters a feature, only the mask records it.
        values = values.masked_fill(~valid, 0)
        # Conservative cell validity: every co-located sample of the packed cell
        # (four CFA phases, or the three co-located pseudo-RAW samples) must exist.
        return values, valid.all(2, keepdim=True)

    def _masked_pool(
        self, features: torch.Tensor, cell_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deterministic masked average pooling onto the token grid.

        ``adaptive_avg_pool2d`` uses the same region for numerator and mask, so
        the ratio is the mean over *valid* feature cells; an all-invalid region
        keeps a zero token and a False mask.
        """
        grid = int(self.config.pool_grid)
        height, width = int(features.shape[-2]), int(features.shape[-1])
        if height < grid or width < grid:
            raise ValueError(
                f"feature map {(height, width)} is smaller than the {grid}x{grid} token grid"
            )
        weight = cell_mask.to(features.dtype)
        region_sum = F.adaptive_avg_pool2d(features * weight, (grid, grid))
        region_fraction = F.adaptive_avg_pool2d(weight, (grid, grid))
        present = region_fraction > 0
        pooled = region_sum / torch.where(
            present, region_fraction, torch.ones_like(region_fraction)
        )
        pooled = pooled.masked_fill(~present, 0)
        batch, channels = pooled.shape[0], pooled.shape[1]
        values = pooled.reshape(batch, channels, grid * grid).transpose(1, 2)
        mask = present.reshape(batch, grid * grid)
        positions = pooled_token_positions(grid, device=features.device, dtype=features.dtype)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)
        return values, mask, positions

    def _resolve_camera(self, frame: MeasurementFrame, state: FullState) -> str:
        camera = frame.camera_id or ""
        if camera:
            if camera not in state.camera_order:
                raise ValueError(
                    f"unknown camera {camera!r}; state holds {list(state.camera_order)}"
                )
            return camera
        if len(state.camera_order) == 1:
            return state.camera_order[0]
        raise ValueError("frame.camera_id is required when the state holds more than one camera")

    def _episode_reset_mask(self, state: FullState, camera: str, episode_ids: Any) -> torch.Tensor:
        ids = _episode_values(episode_ids, state.batch_size)
        previous = state.episodes.get(camera) or [None] * state.batch_size
        flags = [
            prev is not None and prev != current
            for prev, current in zip(previous, ids, strict=True)
        ]
        return torch.tensor(flags, dtype=torch.bool, device=state.device)

    def _unique_flags(
        self, camera_state: Any, frame: MeasurementFrame, batch: int, device: torch.device
    ) -> torch.Tensor:
        """Mirror ``state.push``'s duplicate rule for the real accepted arrival count."""
        new = frame.new_measurement
        if new is None:
            unique = torch.ones(batch, dtype=torch.bool, device=device)
        else:
            unique = torch.as_tensor(new, device=device)
            if unique.ndim == 0:
                unique = unique.expand(batch)
            unique = unique.to(torch.bool).reshape(-1)
        frame_ids = frame.frame_ids
        if frame_ids is not None:
            ids = torch.as_tensor(frame_ids, device=device).to(torch.int64).reshape(-1)
            if ids.numel() == 1 and batch > 1:
                ids = ids.expand(batch)
            if ids.numel() != batch:
                raise ValueError(f"frame_ids must list {batch} entries, got {ids.numel()}")
            duplicate = (
                (camera_state.frame_ids == ids[None]) & camera_state.observed & (ids[None] >= 0)
            ).any(0)
            unique = unique & ~duplicate
        if unique.numel() != batch:
            raise ValueError(f"new_measurement must list {batch} entries, got {unique.numel()}")
        return unique

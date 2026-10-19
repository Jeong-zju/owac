"""N02 v1.2 inference core: joint multi-camera candidate explanations and the RAW objective.

The module is written so that the main chain can be read line by line. Each line of the
P2 chain lands on one function here:

```text
read current + causal history, unique measurement IDs, existing state -> step()
Base multi-scale features with real coordinates + validity          -> encoder(observation)
SAVi-sourced predictor/corrector proposes K joint multi-camera candidates -> _renew_proposals()
camera/time conditioned RAW decoder predicts the same measurement space -> _decode()
noise/validity cost on the same measurement set, q=softmax(log_p0-E/tau) -> _energies() / _support()
learned corrector proposes a state update, accepted only when the
    complete objective does not increase                                -> step() round loop
handle alternative explanations; rebuild real state + readout cache  -> rebuild()
direct spatial features + K states + support enter the Perceiver readout -> readout()
one bridge generates the pi0.5 action chunk; history updates         -> _commit_state()
```

Design commitments (deliberately narrow, matching plan v1.2 sections 4-6):

* ``K`` is a **joint** explanation over all cameras. One candidate decodes every camera; the
  module never picks a per-camera MAP first and then glues mutually inconsistent worlds.
* Proposals come from observed features, learnable latent seeds and the causal prior state,
  refreshed by residuals. Nothing here reads hidden truth, depth, ground-truth boxes or future
  expert actions.
* The declared prior really enters ``log_p0`` and therefore ``q``. Missing, negative or
  non-normalisable priors raise instead of silently falling back to uniform.
* ``q`` is a **support weight**. With a fixed finite candidate set, fixed measurement set and
  ``tau > 0``, ``q*(h) = p0(h) exp(-E(h)/tau) / sum_g p0(g) exp(-E(g)/tau)`` solves
  ``argmin_q E_q[E] + tau KL(q || p0)``; that is the only property claimed. A learned energy is
  not a calibrated posterior.
* A learned corrector only *proposes*: a row accepts the new state only when the complete
  objective ``J = -tau logsumexp(log_p0 - E/tau)`` does not increase. Rejected rows keep the
  previous state and the rejection is recorded. No global convergence is claimed.
* The objective is defined on the **unique** measurement set of the decision time. A repeated
  measurement ID does not add evidence, and the four inner rounds reuse one target set.

Gradient boundary (recorded per step in ``diagnostics``): encoder, decoder, measurement energy,
corrector/predictor and readout are differentiable end to end; the hard candidate *selection*
(``argmax`` over support) and the accept/reject decision are not, and no gradient is claimed for
the search itself. Nothing detaches the encoder before the loss.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from owac.core.encoder_contract_n02 import MeasurementFrame
from owac.frameworks.n02_full import state as state_mod
from owac.frameworks.n02_full.components.savi import SlotAttentionMasked, TransformerPredictor
from owac.frameworks.n02_full.state import ObservationBundle, SequenceState

if TYPE_CHECKING:  # pragma: no cover - annotation only; encoder_v12 is owned by the parallel agent
    from owac.frameworks.n02_full.encoder_v12 import SpatialV12

__all__ = [
    "CHECK_STATUSES",
    "DEFAULT_K",
    "DEFAULT_ROUNDS",
    "DIFFERENTIABLE_PATH",
    "NON_DIFFERENTIABLE_PATH",
    "N02V12Inference",
    "N02V12InferenceError",
    "N02V12NonFiniteError",
    "N02V12StepOutput",
    "QuerySet",
    "V12HistoryEntry",
    "V12InferenceInput",
    "V12RowHistory",
    "build_query_sets",
    "coerce_inference_input",
    "empty_v12_state",
]

DEFAULT_K = 4  # joint explanations, not one MAP per camera
DEFAULT_ROUNDS = 4  # inner loop rounds per decision time
MAX_CAMERAS = 6  # camera identity embedding table (v12 platforms use 2-3 cameras)
CHECK_STATUSES = ("compatible", "insufficient_information", "model_inadequate")
INVALID_VERDICT = "invalid"  # explicit marker; a missing verdict is never the string "None"
DIFFERENTIABLE_PATH = ("encoder", "decoder", "measurement_energy", "corrector_predictor", "readout")
NON_DIFFERENTIABLE_PATH = ("candidate_argmax_selection", "accept_reject_decision")
_PRIOR_FLOOR = 1e-12
_VARIANCE_FLOOR = 1e-12


class N02V12InferenceError(RuntimeError):
    """Wiring error in the v1.2 inference chain; never swallowed silently."""


class N02V12NonFiniteError(N02V12InferenceError):
    """A tensor that must be finite (features, prediction, energy, support) is not."""


# ---------------------------------------------------------------------------------------
# episode state payload
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class V12HistoryEntry:
    """One retained measurement of an episode, as the real features the scan can replay.

    ``tokens``/``valid`` are the corrector's joint multi-camera inputs for that frame. They are
    kept detached because they are *state*, not a live graph: the current step's encoder output
    stays fully differentiable until the loss.
    """

    frame_index: int
    measurement_ids: tuple[str, ...]
    support_hash: str
    tokens: torch.Tensor
    valid: torch.Tensor
    inference_input: Any = None

    def equals(self, other: object) -> bool:
        if not isinstance(other, V12HistoryEntry):
            return False
        return (
            self.frame_index == other.frame_index
            and self.measurement_ids == other.measurement_ids
            and self.support_hash == other.support_hash
            and torch.equal(self.tokens, other.tokens)
            and torch.equal(self.valid, other.valid)
        )


@dataclass(slots=True)
class V12RowHistory:
    """Episode-scoped history plus the readout/attention cache of one batch row."""

    entries: tuple[V12HistoryEntry, ...] = ()
    predicted: torch.Tensor | None = None  # causal prior state (predictor output) of the last step
    tokens: torch.Tensor | None = None  # readout token cache of the last frame
    attention: torch.Tensor | None = None  # readout attention cache of the last frame
    cache_version: int = 0
    spatial: Any = None  # previous observation only; owned and detached

    def zero_(self) -> V12RowHistory:
        """Cleared copy; named ``zero_`` so ``SequenceState.reset_batch`` can clear this row."""
        return V12RowHistory(cache_version=self.cache_version + 1)

    def equals(self, other: object) -> bool:
        if not isinstance(other, V12RowHistory):
            return False
        if len(self.entries) != len(other.entries):
            return False
        if not all(a.equals(b) for a, b in zip(self.entries, other.entries, strict=True)):
            return False
        return (
            self.cache_version == other.cache_version
            and _tensor_equal(self.predicted, other.predicted)
            and _tensor_equal(self.tokens, other.tokens)
            and _tensor_equal(self.attention, other.attention)
        )


def _tensor_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return torch.equal(left, right)


@dataclass(slots=True)
class V12InferenceInput:
    """Everything one decision time may use: measurements, IDs, declared prior, frame index."""

    observation: ObservationBundle
    declared_prior: Any = None
    frame_index: Any = None
    measurement_ids: Any = None


def coerce_inference_input(value: Any) -> V12InferenceInput:
    """Accept the three shapes the harness may hand over, without a compatibility maze.

    Supported: :class:`V12InferenceInput`, :class:`state.InferenceInput` (``bundle`` +
    ``declared_prior``) and a data-v1.2 batch exposing ``observation``/``declared_prior``/
    ``frame_index``/``measurement_ids``.
    """
    if isinstance(value, V12InferenceInput):
        return value
    if isinstance(value, state_mod.InferenceInput):
        return V12InferenceInput(observation=value.bundle, declared_prior=value.declared_prior)
    bundle = getattr(value, "observation", None) or getattr(value, "bundle", None)
    if not isinstance(bundle, ObservationBundle):
        raise N02V12InferenceError(
            "inference input must provide an ObservationBundle as .observation/.bundle, got "
            f"{type(value).__name__}"
        )
    return V12InferenceInput(
        observation=bundle,
        declared_prior=getattr(value, "declared_prior", None),
        frame_index=getattr(value, "frame_index", None),
        measurement_ids=getattr(value, "measurement_ids", None),
    )


# ---------------------------------------------------------------------------------------
# the unique measurement set of one decision time
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class QuerySet:
    """The measurement set of one decision time, in sensor units.

    ``coords`` is ``(B, Q, 2)`` sensor pixels; ``camera_index`` and ``query_channel`` are
    ``(B, Q)`` long tensors indexing :attr:`camera_ids` and the sensor channel (the channel is
    only sent when the decoder declares ``channel_reduction="select"``); ``target`` is the
    observed ``(B, Q)`` value in that declared reduction; ``ids`` is the unique measurement ID
    set that defines the objective for each row.
    """

    coords: torch.Tensor
    camera_index: torch.Tensor
    query_channel: torch.Tensor
    target: torch.Tensor
    masks: dict[str, torch.Tensor]
    ids: tuple[tuple[str, ...], ...]
    camera_ids: tuple[str, ...]
    sampled_per_camera: tuple[int, ...]
    coverage: tuple[float, ...]
    duplicates_dropped: tuple[int, ...]
    channel_reduction: str = "mean"
    per_camera_query_count: tuple[int, ...] = ()


def _frame_values(frame: MeasurementFrame, batch: int) -> torch.Tensor:
    values = torch.as_tensor(frame.values)
    if values.ndim == 4:
        if int(values.shape[0]) != batch:
            raise N02V12InferenceError(
                f"camera {frame.camera_id!r} has batch {values.shape[0]}, expected {batch}"
            )
    elif values.ndim == 3:
        values = values.unsqueeze(0).expand(batch, *values.shape)
    elif values.ndim == 2:
        values = values.reshape(1, 1, *values.shape).expand(batch, 1, *values.shape)
    else:
        raise N02V12InferenceError(f"unsupported frame shape {tuple(values.shape)}")
    if not values.is_floating_point():
        # An integer RAW payload is cast to FP32 as-is, never rescaled here: the loader owns the
        # declared units (LIBERO already divides by 255, RoboTwin keeps its float32 scale).
        values = values.to(torch.float32)
    return values


def _flat_mask(
    frame: MeasurementFrame,
    name: str,
    batch: int,
    size: int,
    *,
    default: bool,
    channel: int | None = None,
) -> torch.Tensor:
    """The named condition, ``(B, H*W)``.

    ``valid_mask`` is true-for-valid, while ``bad_mask``/``missing_mask``/``saturated_mask`` are
    true-for-affected, so an absent mask takes the caller's declared default: valid, or not
    affected. Treating an absent ``bad_mask`` as "everything is bad" would mark every measurement
    invalid.
    """
    raw = getattr(frame, name, None)
    if raw is None:
        return torch.full((batch, size), bool(default), dtype=torch.bool)
    mask = torch.as_tensor(raw).to(torch.bool)
    if mask.ndim == 4:
        mask = (
            mask[:, channel]
            if channel is not None
            else (mask.all(dim=1) if name == "valid_mask" else mask.any(dim=1))
        )
    if mask.ndim == 3:
        if int(mask.shape[0]) != batch:
            raise N02V12InferenceError(
                f"camera {frame.camera_id!r} {name} batch {mask.shape[0]} != {batch}"
            )
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0).expand(batch, *mask.shape)
    else:
        raise N02V12InferenceError(f"unsupported {name} shape {tuple(mask.shape)}")
    flat = mask.reshape(batch, -1)
    if int(flat.shape[-1]) != size:
        raise N02V12InferenceError(
            f"camera {frame.camera_id!r} {name} has {flat.shape[-1]} cells, expected {size}"
        )
    return flat


def build_query_sets(
    observation: ObservationBundle,
    *,
    stride: int = 8,
    max_queries_per_camera: int = 8192,
    declared_ids: tuple[tuple[str, ...], ...] = (),
    channel_reduction: str = "mean",
    measurement_channels: int | None = None,
    input_mode: str = "raw3",
    device: torch.device | None = None,
) -> QuerySet:
    """Build a disclosed sensor target set; stride=1 and cap=0 score every channel/cell.

    Dense primary configs retain every channel. Explicit sparse diagnostic settings
    use evenly distributed raster locations and report their sensor-relative coverage;
    they do not claim to score the unqueried cells.
    """
    if channel_reduction not in ("mean", "select"):
        raise N02V12InferenceError(f"unknown channel_reduction {channel_reduction!r}")
    if stride < 1:
        raise N02V12InferenceError(f"stride must be >= 1, got {stride}")
    if not observation.frames:
        raise N02V12InferenceError("observation bundle has no camera frames")
    batch = _bundle_batch(observation)
    camera_ids = tuple(observation.frames)
    declared = _declared_ids(declared_ids, batch, camera_ids)
    coords_parts: list[torch.Tensor] = []
    camera_parts: list[torch.Tensor] = []
    channel_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    masks = {"valid": [], "bad": [], "missing": [], "saturated": []}
    ids: list[list[str]] = [[] for _ in range(batch)]
    sampled: list[int] = []
    coverage: list[float] = []
    duplicates = [0] * batch
    seen_units = [set() for _ in range(batch)]
    for camera_slot, camera_id in enumerate(camera_ids):
        frame = observation.frames[camera_id]
        values = _frame_values(frame, batch)
        n_channels, height, width = int(values.shape[1]), int(values.shape[2]), int(values.shape[3])
        flat = values.reshape(batch, n_channels, height * width)
        count = max(1, (height * width + stride - 1) // stride)
        if max_queries_per_camera > 0:
            count = min(count, max_queries_per_camera)
        # Include the entire image extent when explicitly requesting a sparse diagnostic.
        # This is still sparse; coverage below is relative to the real sensor, not this set.
        pixels = torch.linspace(0, height * width - 1, count, device=flat.device).round().long()
        if pixels.numel() == 0:
            continue
        rows = torch.div(pixels, width, rounding_mode="floor")
        cols = pixels - rows * width
        frame_ids = tuple(str(item) for item in frame.frame_ids)
        channels = int(measurement_channels or n_channels)
        if channels > n_channels:
            raise N02V12InferenceError(
                f"camera {camera_id!r} has {n_channels} planes but the decoder declares {channels}"
            )
        if channel_reduction == "mean":
            reduced = flat[:, :channels].mean(dim=1)
            selected_channels = (None,)
        else:
            selected_channels = tuple(range(channels))
        unit_ids = _measurement_unit_ids(observation, frame, camera_slot, batch, declared)
        fresh_rows: list[bool] = []
        for row in range(batch):
            unit = unit_ids[row]
            if unit in seen_units[row]:
                # This measurement unit is already in the step's target set: repeating an ID must
                # not add evidence (plan 6.2), so its queries stay invalid and are not listed.
                duplicates[row] += 1
                fresh_rows.append(False)
            else:
                seen_units[row].add(unit)
                ids[row].append(
                    f"{unit}@{height}x{width}/n{int(pixels.numel())}/s{stride}"
                    f"/{frame_ids[row] if row < len(frame_ids) else ''}"
                )
                fresh_rows.append(True)
        fresh = torch.tensor(fresh_rows, dtype=torch.bool, device=flat.device).unsqueeze(-1)
        sampled.append(int(pixels.numel()))
        coverage.append(float(pixels.numel()) / float(height * width))
        for channel in selected_channels:
            channel_masks = {
                name: _flat_mask(
                    frame,
                    f"{name}_mask",
                    batch,
                    height * width,
                    default=name == "valid",
                    channel=channel,
                )[:, pixels]
                for name in masks
            }
            channel_masks["valid"] = channel_masks["valid"] & fresh
            from owac.frameworks.n02_full.observation import (
                _crop_offset,
                packed_to_sensor_coordinates,
            )

            coords_parts.append(
                packed_to_sensor_coordinates(
                    torch.stack([rows, cols], dim=-1),
                    mode=input_mode,
                    crop_offset=_crop_offset(frame),
                ).to(torch.float32)
            )
            camera_parts.append(torch.full((pixels.numel(),), camera_slot, dtype=torch.long))
            index = (
                torch.zeros((pixels.numel(),), dtype=torch.long)
                if channel is None
                else torch.full((pixels.numel(),), int(channel), dtype=torch.long)
            )
            channel_parts.append(index)
            target_parts.append((reduced if channel is None else flat[:, int(channel)])[:, pixels])
            for name in masks:
                masks[name].append(channel_masks[name])
    if not coords_parts:
        raise N02V12InferenceError("measurement set is empty; no sensor cell was sampled")
    coords = torch.cat(coords_parts, dim=0).unsqueeze(0).expand(batch, -1, -1)
    camera_index = torch.cat(camera_parts, dim=0).unsqueeze(0).expand(batch, -1)
    query_channel = torch.cat(channel_parts, dim=0).unsqueeze(0).expand(batch, -1)
    target = torch.cat(target_parts, dim=1)
    query_masks = {name: torch.cat(parts, dim=1) for name, parts in masks.items()}
    camera_index = torch.cat(camera_parts, dim=0)
    counts = _per_camera_query_counts(camera_index, camera_ids)
    if device is not None:
        coords = coords.to(device)
        camera_index = camera_index.to(device)
        query_channel = query_channel.to(device)
        target = target.to(device)
        query_masks = {name: value.to(device) for name, value in query_masks.items()}
    return QuerySet(
        coords=coords,
        camera_index=camera_index.unsqueeze(0).expand(batch, -1),
        query_channel=query_channel,
        target=target,
        masks=query_masks,
        ids=tuple(tuple(items) for items in ids),
        camera_ids=camera_ids,
        sampled_per_camera=tuple(sampled),
        coverage=tuple(coverage),
        duplicates_dropped=tuple(duplicates),
        channel_reduction=channel_reduction,
        per_camera_query_count=counts,
    )


def _per_camera_query_counts(
    camera_index: torch.Tensor, camera_ids: Sequence[str]
) -> tuple[int, ...]:
    """Queries contributed by each camera, in the concatenated query order."""
    return tuple(int((camera_index == index).sum()) for index in range(len(camera_ids)))


def _measurement_unit_ids(
    observation: ObservationBundle,
    frame: MeasurementFrame,
    camera_slot: int,
    batch: int,
    declared: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    """One measurement-unit ID per row: the declared camera ID when present, else a derived one."""
    episode_ids = tuple(str(item) for item in observation.episode_ids)
    frame_ids = tuple(str(item) for item in frame.frame_ids)
    units = []
    for row in range(batch):
        declared_row = declared[row] if row < len(declared) else ()
        if camera_slot < len(declared_row) and str(declared_row[camera_slot]):
            units.append(str(declared_row[camera_slot]))
            continue
        episode = episode_ids[row] if row < len(episode_ids) else ""
        frame_id = frame_ids[row] if row < len(frame_ids) else str(row)
        units.append(f"{episode}/{frame.camera_id}/{frame_id}")
    return tuple(units)


def _declared_ids(
    value: Any, batch: int, camera_ids: Sequence[str] = ()
) -> tuple[tuple[str, ...], ...]:
    """Normalise declared measurement IDs into one tuple per row.

    Two declared forms are accepted: the per-row form (one entry per camera, which is the shape
    inference uses itself) and the ``(episode, camera, frame)`` triples of
    :class:`data_v12.V12WindowBatch.measurement_ids`, which name one measurement per camera of the
    decision time and are therefore mapped by camera ID.
    """
    if value is None or (isinstance(value, (tuple, list)) and not value):
        return tuple(() for _ in range(batch))
    if isinstance(value, str):
        raise N02V12InferenceError("measurement_ids must be a per-row sequence, got one string")
    rows = list(value)
    if (
        rows
        and camera_ids
        and len(rows) == len(camera_ids)
        and all(isinstance(item, (list, tuple)) and len(item) == 3 for item in rows)
    ):
        by_camera = {str(item[1]): f"{item[0]}/{item[1]}/{item[2]}" for item in rows}
        return tuple(
            tuple(by_camera.get(str(camera), "") for camera in camera_ids) for _ in range(batch)
        )
    if len(rows) != batch:
        raise N02V12InferenceError(f"measurement_ids declares {len(rows)} rows, bundle has {batch}")
    return tuple(
        (row,) if isinstance(row, str) else tuple(str(item) for item in row) for row in rows
    )


def _bundle_batch(observation: ObservationBundle) -> int:
    if observation.episode_ids:
        return len(observation.episode_ids)
    for frame in observation.frames.values():
        values = torch.as_tensor(frame.values)
        if values.ndim == 4:
            return int(values.shape[0])
    return 1


# ---------------------------------------------------------------------------------------
# step output
# ---------------------------------------------------------------------------------------


@dataclass(slots=True)
class N02V12StepOutput:
    """One decision time. ``tokens`` is what the pi0.5 feature bridge consumes."""

    q: torch.Tensor  # (B,K) support weights (not a calibrated posterior)
    log_p0: torch.Tensor  # (B,K) log of the normalised declared prior
    energies: torch.Tensor  # (B,K) complete objective per candidate on one measurement set
    rounds: (
        tuple  # per round: {round, accepted, energy_before, energy_after, energies_before/after}
    )
    check: Any  # checks_v12.AbsoluteCheckResult: real status + valid flag
    mean: torch.Tensor  # (B,K,Q) sensor-domain prediction
    variance: torch.Tensor  # (B,K,Q) positive predictive variance
    tokens: torch.Tensor  # (B, n_cameras*256, 2048) readout tokens
    state: SequenceState
    diagnostics: dict = field(default_factory=dict)
    causal: dict | None = None


def empty_v12_state(
    *,
    batch: int = 1,
    k: int = DEFAULT_K,
    n_slots: int = 16,
    slot_dim: int = 512,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> SequenceState:
    """Fresh episode-wide state: zero slots, zero support, empty history/cache per row."""
    return SequenceState(
        episode_ids=tuple("" for _ in range(batch)),
        slot_hidden=torch.zeros((batch, k, n_slots, slot_dim), device=device, dtype=dtype),
        existence=torch.zeros((batch, k), device=device, dtype=dtype),
        geometry=[V12RowHistory() for _ in range(batch)],
        version=0,
    )


class N02V12Inference(nn.Module):
    """The v1.2 loop: joint candidates -> RAW readings -> support -> accepted correction."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        readout: nn.Module,
        measurement: Any,
        n_slots: int = 16,
        slot_dim: int = 512,
        tau: float = 1.0,
        rounds: int = DEFAULT_ROUNDS,
        k: int = DEFAULT_K,
        d_model: int = 512,
        evidence_dim: int | None = None,
        action_dim: int = 0,
        history: int = 8,
        query_stride: int = 1,
        max_queries_per_camera: int = 0,
        calibrator: Any = None,
        strict_check: bool = True,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise N02V12InferenceError(f"tau must be > 0, got {tau}")
        if rounds < 1:
            raise N02V12InferenceError(f"rounds must be >= 1, got {rounds}")
        if k < 2:
            raise N02V12InferenceError(
                f"k must be >= 2 so two explanations can be compared, got {k}"
            )
        if slot_dim % 4:
            raise N02V12InferenceError(f"slot_dim must be divisible by 4, got {slot_dim}")
        self.encoder = encoder
        self.decoder = decoder
        self.readout = readout
        self.measurement = measurement
        self.n_slots = int(n_slots)
        self.slot_dim = int(slot_dim)
        self.tau = float(tau)
        self.rounds = int(rounds)
        self.k = int(k)
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        self.history = int(history)
        self.query_stride = int(query_stride)
        self.max_queries_per_camera = int(max_queries_per_camera)
        self.calibrator = calibrator
        self.strict_check = bool(strict_check)
        self._last_state: SequenceState | None = None

        # Proposal source (P2 "new layers", disclosed): K learnable latent seeds, one per joint
        # explanation, plus learned projections of the observed evidence, of the causal prior
        # state and of the residual feedback.
        self.slot_latents = nn.Parameter(torch.randn(self.k, self.n_slots, self.slot_dim) * 0.02)
        if evidence_dim is not None:
            self.evidence_proj: nn.Module = nn.Linear(
                int(evidence_dim) + self.action_dim, self.slot_dim
            )
        else:
            self.evidence_proj = nn.Identity()  # built lazily from the first observed width
        self.prior_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.residual_proj = nn.Linear(3, self.slot_dim)
        self.renewal = nn.Linear(2 * self.slot_dim, self.slot_dim)
        self.camera_embed = nn.Embedding(MAX_CAMERAS, self.d_model)
        nn.init.zeros_(self.camera_embed.weight)
        # SAVi-sourced corrector/predictor (P0 port of savi.modules.video.Processor pieces).
        self.corrector = SlotAttentionMasked(num_iterations=1, qkv_size=self.slot_dim).build(
            self.slot_dim, self.d_model
        )
        self.predictor = TransformerPredictor(
            width=self.slot_dim, num_heads=4, mlp_size=2 * self.slot_dim, num_layers=2
        )

    # -- public interface -----------------------------------------------------------------

    def step(
        self, inference_input: Any, state: SequenceState, *, emit_readout=True, compute_causal=True
    ) -> N02V12StepOutput:
        """One decision time. Each numbered line below is one line of the P2 chain."""
        inp = coerce_inference_input(inference_input)
        if not isinstance(state, SequenceState):
            raise N02V12InferenceError(f"state must be a SequenceState, got {type(state).__name__}")
        state = self._prepare_state(inp, state)
        batch = _bundle_batch(inp.observation)

        # 1. current measurement set with unique IDs, plus the existing state
        queries = self.build_queries(inp, batch=batch)
        frame_index = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
        # This prediction uses only the retained PREVIOUS observation and state.
        # The current measurement supplies query locations/masks and labels, never features.
        causal = self.causal_prediction(state, queries, frame_index) if compute_causal else None
        # 2. Base multi-scale features with real sensor coordinates and validity
        spatial = self.encoder(inp.observation)
        frame_index = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
        tokens_in = self._corrector_tokens(spatial)
        evidence = self._evidence(spatial, inp.observation)
        device = tokens_in[0].device

        # 3. K joint multi-camera candidate explanations
        slots = self._renew_proposals(
            state.slot_hidden, evidence=evidence, residual_stats=None, tokens=tokens_in
        )
        log_p0, prior_info = self._log_prior(inp, batch=batch, device=device)

        # 4.-6. inner loop on one fixed measurement set: predict the same sensor space, cost it,
        # propose a learned correction and accept it only when the complete objective does not rise.
        mean, variance, valid, decoder_masks = self._decode(spatial, slots, frame_index, queries)
        # Round zero already consumes current image features: it is a fitted diagnostic,
        # separate from the strictly causal prediction built above.
        locked_mean, locked_variance = mean, variance
        energies = self._energies(mean, variance, valid, queries)
        has_measurement = valid.flatten(1).any(dim=-1)
        if torch.is_tensor(state.slot_hidden):
            slots = torch.where(has_measurement[:, None, None, None], slots, state.slot_hidden)
        objective = self._complete_objective(energies, log_p0)
        rounds: list[dict[str, Any]] = []
        accepted_count = torch.zeros(batch, dtype=torch.long, device=device)
        for index in range(self.rounds):
            residuals = self._residuals(queries.target, mean, variance)
            proposal = self._renew_proposals(
                slots,
                evidence=evidence,
                residual_stats=self._residual_stats(residuals, valid),
                tokens=tokens_in,
            )
            mean_p, variance_p, valid_p, _ = self._decode(spatial, proposal, frame_index, queries)
            energies_p = self._energies(mean_p, variance_p, valid_p, queries)
            objective_p = self._complete_objective(energies_p, log_p0)
            keep = has_measurement & (objective_p <= objective)
            rounds.append(
                {
                    "round": index,
                    "accepted": keep.detach(),
                    "energy_before": objective.detach(),
                    "energy_after": objective_p.detach(),
                    "energies_before": energies.detach(),
                    "energies_after": energies_p.detach(),
                }
            )
            accepted_count = accepted_count + keep.to(torch.long)
            slots = torch.where(keep.view(-1, 1, 1, 1), proposal, slots)
            mean = torch.where(keep.view(-1, 1, 1), mean_p, mean)
            variance = torch.where(keep.view(-1, 1, 1), variance_p, variance)
            valid = torch.where(keep.view(-1, 1, 1), valid_p, valid)
            energies = torch.where(keep.view(-1, 1), energies_p, energies)
            objective = torch.where(keep, objective_p, objective)

        # support weights under the declared prior (the only property claimed; see docstring)
        q = self._support(log_p0, energies)

        # 9. real three-state check on the supported explanation's *locked* residuals
        chosen = self._supported_index(q)
        locked_residuals = self._residuals(queries.target, locked_mean, locked_variance)
        check = (
            self._absolute_check(
                causal["residuals"], causal["valid"], queries, causal["chosen"], frame_index
            )
            if causal is not None
            else _invalid_check("no_causal_prefix", self.calibrator)
        )
        fit_residuals = self._residuals(queries.target, mean, variance)

        # 7. direct spatial features + K states + support enter the Perceiver readout; the readout
        # derives its input validity from the SpatialV12 coordinates/validity itself (masks=None).
        if emit_readout:
            tokens, readout_aux = self.readout(
                features=spatial,
                state_tokens=slots,
                support=q,
                masks=None,
            )
        else:
            tokens = slots.new_empty((batch, 0, getattr(self.readout, "out_dim", 2048)))
            readout_aux = {"skipped_prefix_readout": True}
        tokens = _require_tensor(tokens, "readout tokens")
        _assert_finite(tokens, "readout tokens", self.strict_check)

        diagnostics = self._diagnostics(
            queries,
            prior_info,
            q,
            energies,
            objective,
            accepted_count,
            readout_aux,
            decoder_masks,
            fit_residuals,
            locked_residuals,
            chosen,
        )
        diagnostics["prediction_version"] = (
            "causal_pre_locked" if causal is not None else "no_causal_prefix"
        )
        diagnostics["prefit_prediction_version"] = "current_observation_pre_correction"
        diagnostics["saturation_policy"] = "explicit_exclusion_from_primary_objective"
        diagnostics["scored_scalars_per_candidate"] = valid.sum(dim=-1).detach().tolist()
        diagnostics["no_information_rows"] = (~has_measurement).detach().tolist()
        new_state = self._commit_state(inp, state, spatial, queries, slots, q, tokens, readout_aux)
        self._last_state = new_state
        return N02V12StepOutput(
            q=q,
            log_p0=log_p0,
            energies=energies,
            rounds=tuple(rounds),
            check=check,
            mean=mean,
            variance=variance,
            tokens=tokens,
            state=new_state,
            diagnostics=diagnostics,
            causal=causal,
        )

    def causal_prediction(self, state, queries, frame_index):
        """Predict before seeing this frame; candidate selection also uses prior support."""
        means, variances, valids, choices = [], [], [], []
        from owac.frameworks.n02_full.sequence_v12 import subset_queries

        for row, history in enumerate(state.geometry):
            if not isinstance(history, V12RowHistory) or history.spatial is None:
                return None
            selected = subset_queries(queries, torch.tensor([row], device=queries.target.device))
            slots = state.slot_hidden[row : row + 1]
            predicted = self.predictor(slots.flatten(0, 1)).reshape_as(slots)
            mean, variance, valid, _ = self._decode(
                history.spatial, predicted, frame_index[row : row + 1], selected
            )
            means.append(mean)
            variances.append(variance)
            valids.append(valid)
            choices.append(state.existence[row : row + 1].argmax(dim=-1))
        if not means:
            return None
        mean, variance, valid = map(lambda parts: torch.cat(parts), (means, variances, valids))
        return {
            "mean": mean,
            "variance": variance,
            "valid": valid,
            "chosen": torch.cat(choices),
            "support": state.existence.detach(),
            "residuals": self._residuals(queries.target, mean, variance),
            "queries": queries,
        }

    def revise_if_contradicted(self, inference_input, previous, output):
        """Rebuild explanations once after a valid, pre-locked contradiction.

        Raw history is retained. Contradicted slots and derived readout caches
        are not fed into the rebuilt explanations. The original causal verdict
        is retained; fitting the new observation cannot retrospectively pass it.
        Calling the raw inference step below cannot recursively trigger revision.
        """
        check = output.check
        output.diagnostics["revision"] = {"triggered": False, "rows": []}
        if not check.valid or check.status != "model_inadequate":
            return output
        inp = coerce_inference_input(inference_input)
        batch = output.tokens.shape[0]
        if batch == 1:
            rows = [0]
        else:
            per_row = getattr(check, "per_sequence", ())
            if len(per_row) != batch:
                raise N02V12InferenceError("row-specific verdicts required for batched revision")
            rows = [i for i, row in enumerate(per_row) if row["status"] == "model_inadequate"]
        from owac.frameworks.n02_full.sequence_v12 import select_observation

        indices = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
        for row in rows:
            prior = inp.declared_prior
            if prior is not None and torch.as_tensor(prior).ndim == 2:
                prior = prior[row : row + 1]
            one = V12InferenceInput(
                observation=select_observation(inp.observation, [row]),
                declared_prior=prior,
                frame_index=[int(indices[row])],
                measurement_ids=None,
            )
            fresh = empty_v12_state(
                batch=1,
                k=self.k,
                n_slots=self.n_slots,
                slot_dim=self.slot_dim,
                device=output.tokens.device,
            )
            rebuilt = self.step(one, fresh)
            take = torch.tensor([row], device=output.tokens.device)

            def merge_diagnostics(old, new, take=take):
                if isinstance(old, dict) and isinstance(new, dict):
                    return {
                        key: merge_diagnostics(value, new.get(key)) for key, value in old.items()
                    }
                if (
                    torch.is_tensor(old)
                    and torch.is_tensor(new)
                    and old.ndim
                    and new.ndim
                    and old.shape[0] == batch
                    and new.shape[0] == 1
                    and old.shape[1:] == new.shape[1:]
                ):
                    return old.index_copy(0, take, new)
                return old

            output.diagnostics = merge_diagnostics(output.diagnostics, rebuilt.diagnostics)
            if batch == 1:
                output.rounds = rebuilt.rounds
            for name in ("q", "log_p0", "energies", "mean", "variance", "tokens"):
                setattr(
                    output, name, getattr(output, name).index_copy(0, take, getattr(rebuilt, name))
                )
            old_history = output.state.geometry[row]
            new_history = rebuilt.state.geometry[0]
            # Measurements remain available for audit/replay, but all cached
            # explanations of this row come from the new, independent state.
            retained = [
                entry for entry in old_history.entries if entry.frame_index != int(indices[row])
            ]
            new_history.entries = tuple((retained + list(new_history.entries))[-self.history :])
            new_history.cache_version = old_history.cache_version + 1
            geometry = list(output.state.geometry)
            geometry[row] = new_history
            output.state = _replace_state(
                output.state,
                slot_hidden=output.state.slot_hidden.index_copy(0, take, rebuilt.state.slot_hidden),
                existence=output.state.existence.index_copy(0, take, rebuilt.state.existence),
                geometry=geometry,
            )
        output.diagnostics["revision"] = {
            "triggered": bool(rows),
            "rows": rows,
            "budget_per_measurement": 1,
            "kind": "reset_explanations_keep_measurements",
            "causal_verdict_preserved": True,
        }
        self._last_state = output.state
        return output

    def reset(self, mask: Any, state: SequenceState | None = None) -> SequenceState:
        """Row-wise episode reset with :meth:`SequenceState.reset_batch` semantics.

        Selected rows lose slots, support, history and readout/attention caches; unselected rows
        keep their values bit for bit; ``version`` increments once when any row is selected.
        ``state`` defaults to the state returned by the last :meth:`step`.
        """
        target = state if state is not None else self._last_state
        if target is None:
            raise N02V12InferenceError("reset needs a state: pass one or call step() first")
        self._validate_state(target)
        return target.reset_batch(mask)

    def rebuild(
        self, inference_input: Any, state: SequenceState, *, rows: Sequence[int] | None = None
    ):
        """Global recompute of the affected rows from the episode's retained valid history.

        After a measurement was retracted or replaced this recomputes the recurrent state from
        the retained features instead of patching provenance. Real tensors (slots, support) and
        the readout cache are both rewritten. Returns ``(state, diagnostics)``.
        """
        inp = coerce_inference_input(inference_input)
        state = self._prepare_state(inp, state)
        batch = _bundle_batch(inp.observation)
        selected = tuple(range(batch)) if rows is None else tuple(int(row) for row in rows)
        from owac.frameworks.n02_full.sequence_v12 import snapshot

        slots, support = state.slot_hidden.clone(), state.existence.clone()
        geometry = list(state.geometry)
        checks = []
        for row in selected:
            history = geometry[row]
            retained = list(history.entries) if isinstance(history, V12RowHistory) else []
            current = self._snapshot_input(inp, row)
            current_time = int(current.frame_index[0])
            # Replace an existing acquisition at this time; never replay a removed state.
            inputs = [
                entry.inference_input for entry in retained if entry.frame_index != current_time
            ]
            if any(item is None for item in inputs):
                raise N02V12InferenceError("legacy state has no replayable observations; reset it")
            inputs.append(current)
            inputs.sort(key=lambda item: int(item.frame_index[0]))
            fresh = empty_v12_state(
                batch=1,
                k=self.k,
                n_slots=self.n_slots,
                slot_dim=self.slot_dim,
                device=self._device(),
            )
            for item in inputs:
                output = self.step(item, fresh)
                fresh = snapshot(output.state)
            slots[row] = fresh.slot_hidden[0]
            support[row] = fresh.existence[0]
            geometry[row] = fresh.geometry[0]
            checks.append(output.check)
        rebuilt = _replace_state(state, slot_hidden=slots, existence=support, geometry=geometry)
        self._last_state = rebuilt
        return rebuilt, {"rows": selected, "recomputed": True, "from_zero": True, "checks": checks}

    def drop_measurements(
        self, state: SequenceState, measurement_ids: Any, *, rows: Sequence[int] | None = None
    ) -> SequenceState:
        """Remove the named measurements from the episode history (real removal, not provenance)."""
        state = self._validate_state(state)
        wanted = {str(item) for item in _flatten_ids(measurement_ids)}
        geometry = list(state.geometry)
        selected = range(len(geometry)) if rows is None else rows
        for raw_row in selected:
            row = int(raw_row)
            history = geometry[row]
            if not isinstance(history, V12RowHistory):
                continue
            kept = [
                entry for entry in history.entries if not wanted.intersection(entry.measurement_ids)
            ]
            if len(kept) != len(history.entries):
                geometry[row] = V12RowHistory(
                    entries=tuple(kept),
                    predicted=None,  # derived state is invalidated together with the measurement
                    tokens=None,
                    attention=None,
                    cache_version=history.cache_version + 1,
                )
        return _replace_state(state, geometry=list(geometry))

    def snapshot(self, state: SequenceState) -> dict[str, Any]:
        """Serialisable state payload (``torch.save``-able); restore with :meth:`restore`."""
        state = self._validate_state(state)
        return {
            "episode_ids": tuple(state.episode_ids),
            "slot_hidden": _clone(state.slot_hidden),
            "existence": _clone(state.existence),
            "geometry": copy.deepcopy(state.geometry),
            "version": state.version,
        }

    def restore(self, payload: dict[str, Any]) -> SequenceState:
        if not isinstance(payload, dict):
            raise N02V12InferenceError("state payload must be a dict produced by snapshot()")
        return SequenceState(
            episode_ids=tuple(payload.get("episode_ids", ())),
            slot_hidden=_clone(payload.get("slot_hidden")),
            existence=_clone(payload.get("existence")),
            geometry=list(payload.get("geometry", ()) or ()),
            version=int(payload.get("version", 0)),
        )

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameter groups by role, for honest per-path gradient reporting."""
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.decoder.parameters()),
            "readout": list(self.readout.parameters()),
            "proposal": [
                self.slot_latents,
                *self.evidence_proj.parameters(),
                *self.prior_proj.parameters(),
                *self.residual_proj.parameters(),
                *self.renewal.parameters(),
            ],
            "corrector": list(self.corrector.parameters()),
            "predictor": list(self.predictor.parameters()),
            "measurement": list(self.measurement.parameters())
            if isinstance(self.measurement, nn.Module)
            else [],
        }

    # -- internals ------------------------------------------------------------------------

    def _prepare_state(self, inp: V12InferenceInput, state: SequenceState) -> SequenceState:
        state = self._validate_state(state)
        batch = _bundle_batch(inp.observation)
        if state.batch_size is not None and state.batch_size != batch:
            raise N02V12InferenceError(
                f"state batch {state.batch_size} does not match the bundle batch {batch}"
            )
        episode_ids = tuple(str(item) for item in (inp.observation.episode_ids or ()))
        if len(episode_ids) != batch:
            episode_ids = (
                tuple(state.episode_ids)
                if len(state.episode_ids) == batch
                else tuple("" for _ in range(batch))
            )
        previous = tuple(state.episode_ids)
        changed = [
            row for row in range(batch) if previous[row] and episode_ids[row] != previous[row]
        ]
        if changed:
            mask = [row in set(changed) for row in range(batch)]
            state = state.reset_batch(mask)
        return _replace_state(state, episode_ids=episode_ids)

    def _validate_state(self, state: SequenceState) -> SequenceState:
        if not isinstance(state, SequenceState):
            raise N02V12InferenceError(f"state must be a SequenceState, got {type(state).__name__}")
        if state.geometry is None:
            return _replace_state(state, geometry=())
        if not isinstance(state.geometry, (tuple, list)):
            raise N02V12InferenceError(
                "v1.2 state.geometry must hold one V12RowHistory per row; got "
                f"{type(state.geometry).__name__} (a P1 state was passed into the v1.2 path)"
            )
        return state

    def _corrector_tokens(self, spatial: SpatialV12) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint multi-camera tokens for the corrector: one camera-aware token per sensor cell."""
        levels = _get(spatial, "levels") or {}
        valid_map = _get(spatial, "valid") or {}
        level_ids = tuple(_get(spatial, "level_ids") or ())
        camera_ids = tuple(_get(spatial, "camera_ids") or ())
        if not levels or not camera_ids:
            raise N02V12InferenceError("encoder output must expose levels and camera_ids")
        preferred = (
            "stride16" if "stride16" in level_ids else (level_ids[-1] if level_ids else "stride16")
        )
        parts, masks, slots = [], [], []
        for camera_slot, camera_id in enumerate(camera_ids):
            key = (
                f"{camera_id}::{preferred}" if f"{camera_id}::{preferred}" in levels else preferred
            )
            if key not in levels:
                continue
            values = _require_tensor(levels[key], f"feature level {key!r}")
            if values.ndim != 4:
                raise N02V12InferenceError(
                    f"feature level {key!r} must be (B,C,H,W), got {tuple(values.shape)}"
                )
            batch, channels, height, width = values.shape
            tokens = values.reshape(batch, channels, height * width).transpose(1, 2)
            identity = torch.full(
                (tokens.shape[1],),
                camera_slot % MAX_CAMERAS,
                dtype=torch.long,
                device=tokens.device,
            )
            tokens = tokens + self.camera_embed(identity).unsqueeze(0)
            validity = valid_map.get(key)
            valid = (
                torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
                if validity is None
                else torch.as_tensor(validity).reshape(batch, -1).to(torch.bool)
            )
            parts.append(tokens)
            masks.append(valid)
            slots.append(camera_slot)
        if not parts:
            raise N02V12InferenceError(
                f"encoder levels {tuple(levels)} expose no {preferred!r} level for any camera"
            )
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def _evidence(self, spatial: SpatialV12, observation: ObservationBundle) -> torch.Tensor:
        """Observed evidence: pooled multi-scale features (+ the executed action when declared)."""
        levels = _get(spatial, "levels") or {}
        valid_map = _get(spatial, "valid") or {}
        pool: list[torch.Tensor] = []
        for key in sorted(levels):
            values = _require_tensor(levels[key], f"feature level {key!r}")
            if values.ndim != 4:
                continue
            batch, channels = int(values.shape[0]), int(values.shape[1])
            flat = values.reshape(batch, channels, -1)
            validity = valid_map.get(key)
            if validity is None:
                pooled = flat.mean(dim=-1)
            else:
                mask = torch.as_tensor(validity).reshape(batch, -1).to(flat.dtype)
                mass = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                pooled = (flat * mask.unsqueeze(1)).sum(dim=-1) / mass
            pool.append(pooled)
        if not pool:
            raise N02V12InferenceError("encoder output exposes no (B,C,H,W) level to pool")
        observed = torch.cat(pool, dim=-1)
        context = getattr(observation, "context", None)
        if self.action_dim > 0:
            batch = observed.shape[0]
            action = getattr(context, "executed_action", None)
            available = bool(getattr(context, "has_executed_action", False))
            if action is None or not available:
                action = torch.zeros(
                    (batch, self.action_dim), device=observed.device, dtype=observed.dtype
                )
            else:
                action = torch.as_tensor(action).reshape(batch, -1).to(observed.dtype)
                if action.shape[-1] != self.action_dim:
                    raise N02V12InferenceError(
                        f"executed_action has {action.shape[-1]} dims, action_dim {self.action_dim}"
                    )
            observed = torch.cat([observed, action], dim=-1)
        if isinstance(self.evidence_proj, nn.Identity):
            self.evidence_proj = nn.Linear(observed.shape[-1], self.slot_dim).to(observed.device)
        return observed

    def _renew_proposals(
        self,
        slots: torch.Tensor,
        *,
        evidence: torch.Tensor,
        residual_stats: torch.Tensor | None,
        tokens: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """One learned update: refresh the proposal source, then SAVi corrector + predictor."""
        batch, k, n_slots, slot_dim = slots.shape
        latent = self.slot_latents.unsqueeze(0).expand(batch, -1, -1, -1)
        observed = self.evidence_proj(evidence).view(batch, 1, 1, slot_dim)
        renewed = latent + observed
        if bool(torch.is_tensor(slots)) and bool((slots != 0).any()):
            renewed = renewed + torch.tanh(self.prior_proj(slots))
        if residual_stats is not None:
            # residual feedback refreshes the proposal content, not just a category order
            renewed = renewed + self.residual_proj(residual_stats).view(batch, k, 1, slot_dim)
        mixed = self.renewal(torch.cat([slots, renewed], dim=-1))
        features, valid = tokens
        flat = mixed.reshape(batch * k, n_slots, slot_dim)
        expanded = (
            features.unsqueeze(1).expand(batch, k, -1, -1).reshape(batch * k, *features.shape[1:])
        )
        flat_valid = valid.unsqueeze(1).expand(batch, k, -1).reshape(batch * k, valid.shape[-1])
        corrected = self.corrector(flat, expanded, flat_valid)
        predicted = self.predictor(corrected).reshape(batch, k, n_slots, slot_dim)
        _assert_finite(predicted, "candidate slots", self.strict_check)
        return predicted

    def build_queries(self, inp: V12InferenceInput, *, batch: int) -> QuerySet:
        """The measurement set in the decoder's declared output space (see build_query_sets)."""
        return build_query_sets(
            inp.observation,
            stride=self.query_stride,
            max_queries_per_camera=self.max_queries_per_camera,
            declared_ids=inp.measurement_ids,
            channel_reduction=str(getattr(self.decoder, "channel_reduction", "mean")),
            measurement_channels=int(getattr(self.decoder, "measurement_channels", 3)),
            input_mode=str(getattr(self.encoder, "input_mode", "raw3")).lower(),
            device=self._device(),
        )

    def _device(self) -> torch.device:
        """The device the query tensors must live on: wherever this module's parameters are."""
        for parameter in self.parameters():
            return parameter.device
        return torch.device("cpu")

    def _decode(
        self, spatial: SpatialV12, slots: torch.Tensor, frame_index: torch.Tensor, queries: QuerySet
    ):
        """One RAW decode of all K candidates onto the same measurement set.

        The returned ``valid`` is the *data-side* mask (existing measurement, not bad/missing, and
        the decoder's ``in_support`` report). The decoder's predicted ``valid/bad/missing``
        probabilities stay diagnostics: letting a predicted mask shrink the objective would let the
        model hide inconvenient measurements.
        """
        arguments: dict[str, Any] = {
            "features": spatial,
            "slots": slots,
            "frame_index": frame_index,
            "query_coords": queries.coords,
            "query_camera_index": queries.camera_index,
        }
        if queries.channel_reduction == "select":
            arguments["query_channel"] = queries.query_channel
        if getattr(self, "checkpoint_decoder", False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            result = checkpoint(self.decoder, use_reentrant=False, **arguments)
        else:
            result = self.decoder(**arguments)
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise N02V12InferenceError("decoder must return (mean, variance, masks)")
        mean = _require_tensor(result[0], "decoder mean")
        variance = _require_tensor(result[1], "decoder variance")
        decoder_masks = result[2]
        if mean.shape != variance.shape:
            raise N02V12InferenceError(
                f"decoder mean {tuple(mean.shape)} != variance {tuple(variance.shape)}"
            )
        if not isinstance(decoder_masks, dict):
            raise N02V12InferenceError("decoder masks must be a dict of query-level masks")
        _assert_finite(mean, "decoder mean", self.strict_check)
        _assert_finite(variance, "decoder variance", self.strict_check)
        if self.strict_check and not bool((variance > 0).all()):
            raise N02V12InferenceError("decoder variance must be > 0 everywhere (RAW contract)")
        valid = (
            queries.masks["valid"] & ~queries.masks["bad"] & ~queries.masks["missing"]
        ).unsqueeze(1)
        # Primary inference uses explicit exclusion, including when a clip level is known.
        # The standalone censored helper is not wired into this Gaussian objective.
        valid = valid & ~queries.masks["saturated"].unsqueeze(1)
        in_support = decoder_masks.get("in_support")
        if in_support is not None:
            valid = valid & torch.as_tensor(in_support).to(torch.bool)
        valid = valid.expand(-1, int(mean.shape[1]), -1).contiguous()
        masks = {str(name): torch.as_tensor(value) for name, value in decoder_masks.items()}
        return mean, variance.clamp_min(_VARIANCE_FLOOR), valid, masks

    def _energies(self, mean, variance, valid, queries: QuerySet) -> torch.Tensor:
        energies = _require_tensor(
            self.measurement.energy(
                mean=mean, variance=variance, masks=valid, target=queries.target
            ),
            "measurement energy",
        )
        if energies.ndim != 2 or energies.shape[-1] != mean.shape[1]:
            raise N02V12InferenceError(
                f"measurement.energy must return (B,K); got {tuple(energies.shape)}"
                f" for K={int(mean.shape[1])}"
            )
        _assert_finite(energies, "measurement energy", self.strict_check)
        return energies

    def _residuals(
        self, target: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        safe_target = torch.where(torch.isfinite(target), target, 0.0)
        return (safe_target.unsqueeze(1) - mean) / variance.clamp_min(_VARIANCE_FLOOR).sqrt()

    def _residual_stats(self, residuals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residuals = torch.where(valid, residuals, 0.0)
        weight = valid.to(residuals.dtype)
        mass = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        absolute = (residuals.abs() * weight).sum(dim=-1) / mass.squeeze(-1)
        squared = (residuals.square() * weight).sum(dim=-1) / mass.squeeze(-1)
        fraction = weight.mean(dim=-1)
        return torch.stack([absolute, squared.clamp_min(0.0).sqrt(), fraction], dim=-1)

    def _log_prior(self, inp: V12InferenceInput, *, batch: int, device: torch.device):
        """Declared prior -> normalised ``log_p0``; a missing or illegal prior raises."""
        declared = inp.declared_prior
        source = "declared"
        context = getattr(inp.observation, "context", None)
        if declared is None and context is not None:
            available = bool(getattr(context, "hypothesis_prior_available", False))
            candidate = getattr(context, "hypothesis_prior", None)
            if candidate is not None and available:
                declared = candidate
                source = "context"
        if declared is None:
            prior = torch.full((batch, self.k), 1.0 / self.k, dtype=torch.float32)
            source = "uniform"
        else:
            prior = torch.as_tensor(declared, dtype=torch.float32)
            if prior.ndim == 1:
                if prior.numel() != self.k:
                    raise N02V12InferenceError(
                        f"declared prior has {prior.numel()} masses for K={self.k}"
                    )
                prior = prior.unsqueeze(0).expand(batch, -1)
            if prior.ndim != 2 or prior.shape != (batch, self.k):
                raise N02V12InferenceError(
                    f"declared prior must be (B,K)=({batch},{self.k}); got {tuple(prior.shape)}"
                )
            if not bool(torch.isfinite(prior).all()):
                raise N02V12InferenceError("declared prior contains NaN/Inf")
            if bool((prior < 0).any()):
                raise N02V12InferenceError("declared prior has negative mass")
            if not bool((prior.sum(dim=-1) > 0).all()):
                raise N02V12InferenceError(
                    "declared prior has zero total mass for at least one row; a legal prior must "
                    "normalise to 1 (P1 set the true candidate's mass to 0)"
                )
        prior = prior.to(device)
        mass = prior.sum(dim=-1, keepdim=True)
        normalised = prior / mass
        zero_mass = normalised == 0
        log_p0 = normalised.masked_fill(zero_mass, 1.0).log().masked_fill(zero_mass, -torch.inf)
        log_p0 = log_p0 - torch.logsumexp(log_p0, dim=-1, keepdim=True)
        info = {
            "source": source,
            "mass": mass.squeeze(-1).detach(),
            "normalised": normalised.detach(),
        }
        return log_p0, info

    def _support(self, log_p0: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
        return torch.softmax(log_p0 - energies / self.tau, dim=-1)

    def _complete_objective(self, energies: torch.Tensor, log_p0: torch.Tensor) -> torch.Tensor:
        """``J = -tau logsumexp(log_p0 - E/tau)``: the complete objective on one measurement set.

        ``min_q E_q[E] + tau KL(q||p0)`` equals ``J``; a state update is accepted only when
        ``J`` does not increase. No convergence claim follows from that.
        """
        return -self.tau * torch.logsumexp(log_p0 - energies / self.tau, dim=-1)

    def _supported_index(self, q: torch.Tensor) -> torch.Tensor:
        return torch.argmax(q.detach(), dim=-1)

    def _absolute_check(self, residuals, valid, queries: QuerySet, chosen, frame_index) -> Any:
        """Three-state check on the supported explanation's locked residuals.

        The prediction and chosen candidate are locked from the preceding state.
        Diagnostic relative time is zero; decoder conditioning retains real source time.
        Missing prefix/calibration produces an invalid result.
        """
        supported = torch.gather(
            residuals, 1, chosen.view(-1, 1, 1).expand(-1, 1, residuals.shape[-1])
        ).squeeze(1)
        supported_valid = torch.gather(
            valid, 1, chosen.view(-1, 1, 1).expand(-1, 1, valid.shape[-1])
        ).squeeze(1)
        groups = self._residual_views(supported, supported_valid, queries, frame_index)
        version = str(getattr(self.calibrator, "version", "n02-v12-windowed-check-v1"))
        try:
            result = self.measurement.absolute_check(
                residuals=groups,
                calibrator=self.calibrator,
                version=version,
                rule=getattr(self.calibrator, "rule", None),
                measurement_ids=tuple(item for row in queries.ids for item in row),
                prediction_version="causal_pre_locked",
            )
        except Exception as exc:  # noqa: BLE001 - a broken check stays visible, never silent
            if self.strict_check:
                raise
            return _invalid_check(
                f"absolute_check raised {type(exc).__name__}: {exc}", self.calibrator
            )
        return _validate_check(result, self.strict_check, calibrator=self.calibrator)

    def _residual_views(
        self,
        residuals: torch.Tensor,
        valid: torch.Tensor,
        queries: QuerySet,
        frame_index: torch.Tensor,
    ):
        """One :class:`ResidualView` group per row: one view per camera, in query order."""
        from owac.frameworks.n02_full.checks import ResidualView

        counts = queries.per_camera_query_count
        groups: list[list[Any]] = []
        for row in range(int(residuals.shape[0])):
            views: list[Any] = []
            cursor = 0
            for slot, camera_id in enumerate(queries.camera_ids):
                count = int(counts[slot]) if slot < len(counts) else 0
                if count <= 0:
                    continue
                stop = cursor + count
                views.append(
                    ResidualView(
                        standardised_residual=residuals[row, cursor:stop].reshape(1, -1),
                        valid=valid[row, cursor:stop].reshape(1, -1),
                        scale=f"sensor_stride{self.query_stride}",
                        frame_index=0,  # relative diagnostic axis; decoder keeps real source time
                        region_id=str(camera_id),
                        origin=(0, 0),
                    )
                )
                cursor = stop
            groups.append(views)
        return groups

    def _diagnostics(
        self,
        queries,
        prior_info,
        q,
        energies,
        objective,
        accepted_count,
        readout_aux,
        decoder_masks,
        fit_residuals,
        locked_residuals,
        chosen,
    ) -> dict:
        with torch.no_grad():
            entropy = -(q.clamp_min(_PRIOR_FLOOR) * q.clamp_min(_PRIOR_FLOOR).log()).sum(dim=-1)
            best = torch.gather(energies, 1, chosen.view(-1, 1)).squeeze(1)
            spread = energies.max(dim=-1).values - energies.min(dim=-1).values
            locked_mean_abs = _masked_mean_abs(locked_residuals, chosen)
            fit_mean_abs = _masked_mean_abs(fit_residuals, chosen)
        return {
            "meas_ids": queries.ids,
            "meas_id_hash": _hash_ids(queries.ids),
            "n_measurements": tuple(len(items) for items in queries.ids),
            "duplicates_dropped": queries.duplicates_dropped,
            "sampled_per_camera": queries.sampled_per_camera,
            "coverage": queries.coverage,
            "query_stride": self.query_stride,
            "channel_reduction": queries.channel_reduction,
            "per_camera_query_count": queries.per_camera_query_count,
            "camera_ids": queries.camera_ids,
            "prior": {"source": prior_info["source"], "mass": prior_info["mass"]},
            "support": q,
            "support_entropy": entropy,
            "objective": objective.detach(),
            "accepted_rounds": accepted_count,
            "rounds": self.rounds,
            "energy_spread": spread,
            "argmax_candidate": chosen,
            "best_energy": best.detach(),
            "readout_aux": dict(readout_aux or {}),
            "decoder_masks": {
                str(name): value.detach() for name, value in dict(decoder_masks or {}).items()
            },
            "locked_residual_mean_abs": locked_mean_abs,
            "fit_residual_mean_abs": fit_mean_abs,
            "differentiable_path": DIFFERENTIABLE_PATH,
            "nondifferentiable_path": NON_DIFFERENTIABLE_PATH,
            "gradient_note": (
                "encoder/decoder/energy/corrector/readout are differentiable end to end; the "
                "argmax candidate selection and the accept/reject decision are hard and carry no "
                "gradient. A gradient on a projection is not a gradient on the search."
            ),
        }

    def _commit_state(
        self, inp, state, spatial, queries, slots, q, tokens, readout_aux
    ) -> SequenceState:
        """Append the accepted measurement, refresh the readout/attention cache, keep the scan."""
        geometry = list(state.geometry)
        batch = int(slots.shape[0])
        features, valid = self._corrector_tokens(spatial)
        frame_index = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
        support_hash = str(_get(spatial, "support_hash") or "")
        attention = readout_aux.get("attention") if isinstance(readout_aux, dict) else None
        if torch.is_tensor(attention):
            attention = attention.detach()
        for row in range(batch):
            history = (
                geometry[row]
                if row < len(geometry) and isinstance(geometry[row], V12RowHistory)
                else V12RowHistory()
            )
            entry = V12HistoryEntry(
                frame_index=int(frame_index[row]),
                measurement_ids=queries.ids[row],
                support_hash=support_hash,
                tokens=features[row].detach().clone(),
                valid=valid[row].detach().clone(),
                inference_input=self._snapshot_input(inp, row),
            )
            geometry[row] = V12RowHistory(
                entries=(history.entries + (entry,))[-max(1, self.history) :],
                predicted=slots[row].detach().clone(),
                tokens=tokens[row].detach().clone() if tokens.shape[1] else None,
                attention=attention,
                cache_version=history.cache_version + 1,
                spatial=self._snapshot_spatial(spatial, row),
            )
        return _replace_state(
            state, slot_hidden=slots, existence=q.detach(), geometry=list(geometry)
        )

    @staticmethod
    def _snapshot_spatial(spatial, row):
        from owac.frameworks.n02_full.sequence_v12 import select_spatial, snapshot

        return snapshot(select_spatial(spatial, row))

    @staticmethod
    def _snapshot_input(inp, row):
        from owac.frameworks.n02_full.sequence_v12 import select_observation, snapshot

        batch = _bundle_batch(inp.observation)
        frames = _frame_index_tensor(inp, observation=inp.observation, batch=batch)
        prior = inp.declared_prior
        if prior is not None and torch.as_tensor(prior).ndim == 2:
            prior = prior[row : row + 1]
        return snapshot(
            V12InferenceInput(
                select_observation(inp.observation, [row]), prior, [int(frames[row])], None
            )
        )


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _require_tensor(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def _clone(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def _replace_state(state: SequenceState, **updates: Any) -> SequenceState:
    payload = {
        "episode_ids": state.episode_ids,
        "slot_hidden": state.slot_hidden,
        "existence": state.existence,
        "geometry": state.geometry,
        "version": state.version,
    }
    payload.update(updates)
    return SequenceState(**payload)


def _assert_finite(value: torch.Tensor, name: str, strict: bool) -> None:
    if not torch.is_tensor(value):
        return
    if not bool(torch.isfinite(value).all()):
        raise N02V12NonFiniteError(f"{name} contains NaN/Inf (strict={strict})")


def _frame_index_tensor(
    inp: V12InferenceInput, *, observation: ObservationBundle, batch: int
) -> torch.Tensor:
    raw = inp.frame_index
    if raw is None:
        values: list[int] = []
        frames = list(observation.frames.values())
        for row in range(batch):
            value = 0
            for frame in frames:
                ids = tuple(frame.frame_ids)
                if row < len(ids):
                    try:
                        value = max(value, int(str(ids[row])))
                    except ValueError:
                        value = max(value, 0)
            values.append(value)
    elif torch.is_tensor(raw):
        values = [int(item) for item in raw.reshape(-1).tolist()]
    elif isinstance(raw, (list, tuple)):
        values = [int(item) for item in raw]
    else:
        values = [int(raw)] * batch
    if len(values) != batch:
        values = (values + [0] * batch)[:batch]
    return torch.tensor(values, dtype=torch.long)


def _hash_ids(ids: Sequence[Sequence[str]]) -> str:
    payload = "\u0001".join("\u0000".join(items) for items in ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _flatten_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    out: list[str] = []
    for item in value:
        out.extend(_flatten_ids(item))
    return tuple(out)


def _validate_check(result: Any, strict: bool, *, calibrator: Any) -> Any:
    """Require a real three-state verdict; a missing verdict is an explicit failure."""
    status = getattr(result, "status", None)
    valid = getattr(result, "valid", None)
    if status is None or valid is None:
        message = (
            "absolute_check must return AbsoluteCheckResult(status, valid, statistics, "
            f"calibration_version); got {type(result).__name__} without a real verdict"
        )
        if strict:
            raise N02V12InferenceError(message)
        return _invalid_check(message, calibrator)
    if str(status) not in CHECK_STATUSES:
        message = (
            f"absolute_check returned unknown status {status!r}; expected one of {CHECK_STATUSES}"
        )
        if strict:
            raise N02V12InferenceError(message)
        return _invalid_check(message, calibrator)
    return result


def _masked_mean_abs(residuals: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Mean |residual| of the supported candidate per row (a diagnostic, not a verdict)."""
    picked = torch.gather(residuals, 1, chosen.view(-1, 1, 1).expand(-1, 1, residuals.shape[-1]))
    return picked.abs().mean(dim=-1).squeeze(1).detach()


def _invalid_check(reason: str, calibrator: Any) -> Any:
    """An explicitly invalid check: never counted as 0 false alarms / 0 detections."""
    from owac.frameworks.n02_full import checks_v12

    return checks_v12.AbsoluteCheckResult.invalid(
        reason=reason,
        calibration_version=_calibrator_version(calibrator),
        statistics={"verdict": INVALID_VERDICT},
    )


def _calibrator_version(calibrator: Any) -> str:
    for name in ("version", "calibration_version", "name"):
        value = getattr(calibrator, name, None)
        if isinstance(value, str) and value:
            return value
    return "uncalibrated"


def _dummy_bundle(batch: int) -> ObservationBundle:
    from owac.core.encoder_contract_n02 import EmbodimentContext

    return ObservationBundle(
        frames={}, context=EmbodimentContext(), episode_ids=tuple("" for _ in range(batch))
    )

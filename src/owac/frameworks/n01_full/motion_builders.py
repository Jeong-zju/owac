"""P2 motion bridges: causal-window offsets for the adapter and redacted-ledger rebuilds.

Two deliverables live here, both thin adapters over the motion-core
:mod:`owac.backends.raw_correspondence` (no second numeric conversion is written):

* :class:`WindowMotionProvider` turns one camera's already-arrived causal window
  (newest first, slot 0 is the reference) into the ``[B, T, 2, H, W]`` packed-cell
  pull offsets the DBSR fusion consumes.  All coordinate work stays inside the
  provider: the estimator speaks full-sensor ``xy`` through
  :class:`~owac.backends.raw_correspondence.RawSensorFrame`, and the fusion reads
  packed cells (one packed cell = ``sensor_per_packed`` sensor pixels; CFA keeps
  its four phase offsets).  The adapter never rewrites a unit.
* :class:`LedgerCorrespondenceBuilder` / :class:`LedgerNoiseModelBuilder` are the
  P2 recomputation callbacks for ``counterfactual_replay``.  They hold an
  estimator, geometry declarations and *identities* only; every value is read from
  the ``MeasurementLedger`` view passed to ``__call__``, so a redacted view
  (retracted samples physically removed, ``without`` zeroes them and marks them
  missing) cannot leak through a captured cache or an un-cleared original.

Row model of :class:`RebuiltCorrespondence`: every row is one valid query of the
reference frame at its real sampling position, and its weights are the bilinear
pull taps of that query summed over the contributing source frames and
renormalised to sum to 1.  A query with no contributing tap is dropped, never
invented.  The spec names its columns (``InterpolationSpec.source_ids``) so the
fit re-indexes them onto the domain's canonical evidence order by identity.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from owac.backends.raw_correspondence import (
    MODE_CHANNELS,
    SENSOR_PER_PACKED,
    CorrespondenceEstimator,
    RawCorrespondenceMap,
    RawSensorFrame,
)
from owac.core.encoder_contract import MeasurementFrame
from owac.frameworks.n01.state import N01State
from owac.frameworks.n01_full.domains import canonical_sources
from owac.frameworks.n01_full.ledger import VALID, MeasurementLedger
from owac.frameworks.n01_full.statistics import InterpolationSpec, NoiseModel
from owac.raw.n01_noise import NoiseCalibration, build_noise_model

__all__ = [
    "CameraCorrespondenceError",
    "LedgerCorrespondenceBuilder",
    "LedgerNoiseModelBuilder",
    "RebuiltCorrespondence",
    "WindowMotion",
    "WindowMotionProvider",
    "raw_frame_from_ledger",
]

#: Declared modes of the provider/builder: the motion-core stems.
MODES = ("cfa4", "rgb3")

_PHASE_NAMES = ("RGGB", "BGGR", "GRBG", "GBRG")


class CameraCorrespondenceError(ValueError):
    """A window/frame does not match the declared motion geometry."""


def _mode_of(cfa_pattern: str) -> str:
    if cfa_pattern in _PHASE_NAMES:
        return "cfa4"
    if cfa_pattern == "rgb3":
        return "rgb3"
    raise CameraCorrespondenceError(
        f"unknown cfa_pattern {cfa_pattern!r}; expected one of {_PHASE_NAMES} or 'rgb3'"
    )


def raw_frame_from_ledger(
    frame: Any,
    *,
    cfa_pattern: str,
    channel_phase: Sequence[int],
    sensor_origin_xy: tuple[int, int],
) -> RawSensorFrame:
    """Rebuild one :class:`RawSensorFrame` from a stored ledger frame.

    Only identity, geometry declarations and the *stored* values/validity are
    used; a retracted sample is already zeroed and marked missing inside the
    view, so it enters as an invalid sample and never reaches a tap.
    """
    frame_id = frame.identity.frame_id
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise CameraCorrespondenceError(
            "the P2 motion builder needs integer exposure ids; "
            f"got {frame_id!r} for {tuple(frame.identity)}"
        )
    validity = frame.validity == VALID
    values = frame.values.to(torch.float32)
    return RawSensorFrame(
        dataset_id=frame.identity.dataset_id,
        episode_id=frame.identity.episode_id,
        camera_id=frame.identity.camera_id,
        frame_id=frame_id,
        values=values,
        channel_phase=tuple(int(item) for item in channel_phase),
        cfa_pattern=cfa_pattern,
        sensor_origin_xy=(int(sensor_origin_xy[0]), int(sensor_origin_xy[1])),
        validity=validity,
        time=float(frame.timestamp),
        calibration_version=frame.calibration_version or "unversioned",
    )


def _sensor_frame_from_state(
    state: N01State,
    slot: int,
    member: int,
    *,
    dataset_id: str,
    episode_id: str,
    cfa_pattern: str,
    channel_phase: Sequence[int],
    sensor_origin_xy: tuple[int, int],
    calibration_version: str,
) -> RawSensorFrame:
    if not bool(state.observed[slot, member]):
        raise CameraCorrespondenceError(
            f"slot {slot} of member {member} holds no unique measurement"
        )
    frame_id = int(state.frame_ids[slot, member])
    if frame_id < 0:
        raise CameraCorrespondenceError(f"slot {slot} of member {member} has no frame identity")
    values = state.values[slot, member].to(torch.float32)
    valid = state.valid[slot, member].to(torch.bool)
    if not bool(valid.any()):
        raise CameraCorrespondenceError(
            f"slot {slot} of member {member} has no valid sample to correspond"
        )
    return RawSensorFrame(
        dataset_id=dataset_id,
        episode_id=episode_id,
        camera_id=state.camera_ids[0],
        frame_id=frame_id,
        values=values.masked_fill(~valid, 0),
        channel_phase=tuple(int(item) for item in channel_phase),
        cfa_pattern=cfa_pattern,
        sensor_origin_xy=(int(sensor_origin_xy[0]), int(sensor_origin_xy[1])),
        validity=valid,
        time=float(state.timestamps[slot, member]),
        calibration_version=calibration_version,
    )


@dataclass
class WindowMotion:
    """One causal window's motion result, in the fusion's own packed-cell units."""

    offsets_xy: torch.Tensor  # [B, T, 2, H, W] packed-cell pull offsets; slot 0 is zero
    valid: torch.Tensor  # [B, T] bool; slot 0 is True (the reference maps to itself)
    mapping_version: str
    decision_version: str
    risk_kind: str = "unavailable"
    risk: torch.Tensor | None = None  # [B, T, H, W] sensor-pixel proxy or calibrated risk
    visibility: torch.Tensor | None = None  # [B, T, H, W]
    reference_frame_id: Any = None
    source_frame_ids: tuple[Any, ...] = ()
    sensor_per_packed: int = 1
    query_shape: tuple[int, int] = (0, 0)

    def describe(self) -> dict[str, Any]:
        return {
            "mapping_version": self.mapping_version,
            "decision_version": self.decision_version,
            "risk_kind": self.risk_kind,
            "reference_frame_id": self.reference_frame_id,
            "source_frame_ids": [
                None if item is None else int(item) for item in self.source_frame_ids
            ],
            "window_frames": int(self.offsets_xy.shape[1]),
            "packed_shape": [int(self.offsets_xy.shape[-2]), int(self.offsets_xy.shape[-1])],
            "sensor_per_packed": int(self.sensor_per_packed),
            "valid_pairs_per_member": [int(value) for value in self.valid.sum(-1).tolist()],
            "offsets_sha256": _tensor_digest(self.offsets_xy),
        }


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(torch.float32).cpu().contiguous()
    digest = hashlib.sha256(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class WindowMotionProvider:
    """Estimate one camera's causal-window pull offsets with the motion-core estimator.

    ``mode`` is implied by ``cfa_pattern`` (``rg gb``-style patterns build the
    ``cfa4`` stem, ``rgb3`` the three-channel stem); ``channel_phase`` and
    ``sensor_origin_xy`` are the packing declarations of
    :class:`~owac.backends.raw_correspondence.RawSensorFrame`.

    The provider is memoryless and only reads ``state`` (the already-arrived
    buffer of that camera) and the arriving ``frame``; it can never reach a future
    exposure.  ``window`` is the exact number of slots the adapter is fusing, so
    the returned ``T`` equals the step's ``window_frames`` by construction.
    """

    def __init__(
        self,
        estimator: CorrespondenceEstimator,
        *,
        dataset_id: str,
        cfa_pattern: str = "rgb3",
        channel_phase: Sequence[int] = (),
        sensor_origin_xy: tuple[int, int] = (0, 0),
        calibration_version: str = "unversioned",
        decision_version: str = "p2-motion-window-v1",
        episode_fallback: str = "unversioned",
        iters: int | None = None,
        taps: bool = False,
    ) -> None:
        if not isinstance(estimator, CorrespondenceEstimator):
            raise TypeError("WindowMotionProvider needs a CorrespondenceEstimator")
        if not dataset_id:
            raise ValueError("WindowMotionProvider.dataset_id must be non-empty")
        mode = _mode_of(cfa_pattern)
        if estimator.mode != mode:
            raise CameraCorrespondenceError(
                f"estimator builds the {estimator.mode} stem but cfa_pattern {cfa_pattern!r} "
                f"needs {mode}"
            )
        if mode == "cfa4" and tuple(channel_phase) == ():
            raise CameraCorrespondenceError(
                "cfa4 needs the channel_phase permutation that maps channels to CFA phases"
            )
        if not calibration_version:
            raise ValueError("calibration_version must be declared")
        self._estimator = estimator
        self.dataset_id = dataset_id
        self.cfa_pattern = cfa_pattern
        self.channel_phase = tuple(int(item) for item in channel_phase)
        self.sensor_origin_xy = (int(sensor_origin_xy[0]), int(sensor_origin_xy[1]))
        self.calibration_version = calibration_version
        self.decision_version = decision_version
        self.episode_fallback = episode_fallback
        self._iters = iters
        self._taps = bool(taps)

    @property
    def version(self) -> str:
        return self.decision_version

    def __call__(
        self,
        frame: MeasurementFrame,
        state: N01State,
        camera_id: str | None = None,
        *,
        window: int | None = None,
    ) -> WindowMotion:
        if not isinstance(state, N01State):
            raise TypeError("WindowMotionProvider expects the camera's N01State")
        camera = camera_id or state.camera_ids[0]
        if camera not in state.camera_ids:
            raise CameraCorrespondenceError(f"camera {camera!r} is not in the state")
        values = frame.values
        if not torch.is_tensor(values) or values.ndim != 4:
            raise CameraCorrespondenceError("frame.values must be a [B, C, H, W] tensor")
        batch, channels, height, width = (int(size) for size in values.shape)
        if batch != state.batch_size:
            raise CameraCorrespondenceError(
                f"frame batch {batch} does not match the state batch {state.batch_size}"
            )
        expected_channels = MODE_CHANNELS[_mode_of(self.cfa_pattern)]
        if channels != expected_channels:
            raise CameraCorrespondenceError(
                f"{self.cfa_pattern} needs {expected_channels} packed channels, got {channels}"
            )
        count = int(state.filled.max()) if window is None else int(window)
        if count < 1:
            raise CameraCorrespondenceError("the window holds no unique measurement")
        if count > state.num_slots:
            raise CameraCorrespondenceError(
                f"declared window {count} exceeds the buffer capacity {state.num_slots}"
            )
        if (height, width) != state.shape:
            raise CameraCorrespondenceError(
                f"frame {(height, width)} does not match the buffer grid {state.shape}"
            )

        device = state.values.device
        offsets = torch.zeros(batch, count, 2, height, width, dtype=torch.float32, device=device)
        valid = torch.zeros(batch, count, dtype=torch.bool, device=device)
        valid[:, 0] = True
        risk: torch.Tensor | None = None
        visibility: torch.Tensor | None = None
        kinds: list[str] = []
        versions: list[str] = []
        source_ids: tuple[Any, ...] = tuple(
            None if int(state.frame_ids[slot, 0]) < 0 else int(state.frame_ids[slot, 0])
            for slot in range(count)
        )
        step = SENSOR_PER_PACKED[_mode_of(self.cfa_pattern)]
        for member in range(batch):
            if not bool(state.observed[0, member]):
                # This member has no unique measurement yet: no reference, no pairs.
                continue
            episode = self.episode_fallback
            if frame.episode_ids is not None:
                episode = str(frame.episode_ids[member])
            reference = _sensor_frame_from_state(
                state,
                0,
                member,
                dataset_id=self.dataset_id,
                episode_id=episode,
                cfa_pattern=self.cfa_pattern,
                channel_phase=self.channel_phase,
                sensor_origin_xy=self.sensor_origin_xy,
                calibration_version=self.calibration_version,
            )
            for slot in range(1, count):
                if not bool(state.observed[slot, member]):
                    # The window is the batch maximum; a member with fewer unique
                    # frames has no pair for its empty slots (and no invented frame).
                    continue
                source = _sensor_frame_from_state(
                    state,
                    slot,
                    member,
                    dataset_id=self.dataset_id,
                    episode_id=episode,
                    cfa_pattern=self.cfa_pattern,
                    channel_phase=self.channel_phase,
                    sensor_origin_xy=self.sensor_origin_xy,
                    calibration_version=self.calibration_version,
                )
                estimated = self._estimator.estimate(
                    reference, source, iters=self._iters, taps=self._taps
                )
                offsets[member, slot] = estimated.packed_flow.to(torch.float32)
                valid[member, slot] = _map_valid(estimated)
                kinds.append(estimated.risk_kind)
                versions.append(estimated.mapping_version)
                if estimated.flow_risk is not None:
                    if risk is None:
                        risk = torch.zeros(
                            batch, count, height, width, dtype=torch.float32, device=device
                        )
                        visibility = torch.zeros_like(risk)
                    risk[member, slot] = estimated.flow_risk.to(torch.float32)
                    if estimated.visibility is not None:
                        visibility[member, slot] = estimated.visibility.to(torch.float32)
        return WindowMotion(
            offsets_xy=offsets,
            valid=valid,
            mapping_version=_uniform(versions, "n01-full-v1"),
            decision_version=self.decision_version,
            risk_kind=_risk_kind(kinds),
            risk=risk,
            visibility=visibility,
            reference_frame_id=source_ids[0] if source_ids else None,
            source_frame_ids=source_ids,
            sensor_per_packed=step,
            query_shape=(height, width),
        )


def _map_valid(estimated: RawCorrespondenceMap) -> bool:
    value = estimated.valid
    if value is False:
        return False
    if torch.is_tensor(value):
        return bool(value.any())
    return bool(value)


def _uniform(values: Sequence[str], default: str) -> str:
    unique = sorted(set(values))
    if not unique:
        return default
    if len(unique) > 1:
        raise CameraCorrespondenceError(f"mixed mapping versions inside one window: {unique}")
    return unique[0]


def _risk_kind(kinds: Sequence[str]) -> str:
    unique = sorted(set(kinds))
    if not unique:
        return "unavailable"
    if len(unique) == 1:
        return unique[0]
    return "mixed(" + ",".join(unique) + ")"


@dataclass(frozen=True)
class RebuiltCorrespondence(InterpolationSpec):
    """A correspondence recomputed from a redacted ledger view, with its provenance.

    It *is* an :class:`InterpolationSpec` (columns named through ``source_ids``),
    so ``fit_structure`` consumes it directly and re-indexes the columns onto the
    domain's canonical evidence order by identity.
    """

    mapping_version: str = ""
    decision_version: str = ""
    risk_kind: str = "unavailable"
    reference_frame_id: Any = None
    source_frame_ids: tuple[Any, ...] = ()
    valid_rows: int = 0
    dropped_rows: int = 0
    risk: torch.Tensor | None = field(default=None, compare=False)

    def describe(self) -> dict[str, Any]:
        sums = self.weights.sum(dim=1)
        return {
            "kind": "rebuilt-correspondence",
            "mapping_version": self.mapping_version,
            "decision_version": self.decision_version,
            "risk_kind": self.risk_kind,
            "reference_frame_id": self.reference_frame_id,
            "source_frame_ids": [
                None if item is None else int(item) for item in self.source_frame_ids
            ],
            "rows": int(self.weights.shape[0]),
            "columns": int(self.weights.shape[1]),
            "valid_rows": int(self.valid_rows),
            "dropped_rows": int(self.dropped_rows),
            "row_weight_sum_min": float(sums.min()) if sums.numel() else 0.0,
            "row_weight_sum_max": float(sums.max()) if sums.numel() else 0.0,
            "risk_mean": None if self.risk is None else float(self.risk.mean()),
            "weights_sha256": _tensor_digest(self.weights),
        }


class LedgerCorrespondenceBuilder:
    """Recompute the P2 pull correspondence from the ledger view alone.

    ``evidence_ids`` declares which samples this correspondence is allowed to
    reference (identities, never content).  On every call the builder:

    1. lists the retained frames of one camera from the *passed* view,
    2. rebuilds their :class:`RawSensorFrame`s from the view's own values and
       validity (a retracted sample is already zero/missing there),
    3. runs the shared estimator for each already-retained source frame against
       the reference frame, and
    4. returns a :class:`RebuiltCorrespondence` whose rows are the reference
       frame's valid queries and whose columns are named by identity.

    It holds no frame tensors, no content cache and no earlier view, so calling it
    on a redacted view cannot see the original bytes.
    """

    def __init__(
        self,
        estimator: CorrespondenceEstimator,
        *,
        evidence_ids: Sequence[Any],
        cfa_pattern: str,
        channel_phase: Sequence[int] = (),
        sensor_origin_xy: tuple[int, int] = (0, 0),
        camera_id: str | None = None,
        reference_frame_id: int | None = None,
        decision_version: str = "p2-ledger-correspondence-v1",
        iters: int | None = None,
    ) -> None:
        if not isinstance(estimator, CorrespondenceEstimator):
            raise TypeError("LedgerCorrespondenceBuilder needs a CorrespondenceEstimator")
        if not evidence_ids:
            raise ValueError("evidence_ids must name at least one sample identity")
        mode = _mode_of(cfa_pattern)
        if estimator.mode != mode:
            raise CameraCorrespondenceError(
                f"estimator builds the {estimator.mode} stem but cfa_pattern {cfa_pattern!r} "
                f"needs {mode}"
            )
        self._estimator = estimator
        self.evidence_ids = tuple(evidence_ids)
        self.cfa_pattern = cfa_pattern
        self.channel_phase = tuple(int(item) for item in channel_phase)
        self.sensor_origin_xy = (int(sensor_origin_xy[0]), int(sensor_origin_xy[1]))
        self.camera_id = camera_id
        self.reference_frame_id = reference_frame_id
        self.decision_version = decision_version
        self._iters = iters
        self.calls = 0

    @property
    def version(self) -> str:
        return self.decision_version

    def usable_columns(self, ledger: MeasurementLedger) -> list[Any]:
        """The declared evidence actually usable in this view, in canonical order.

        This is exactly the filter ``fit_structure`` applies before it builds
        ``used``, so the returned column order matches the fit's.
        """
        ordered, _duplicates = canonical_sources(self.evidence_ids)
        return [
            key for key in ordered if ledger.has_source(key) and ledger.validity_of(key) == "valid"
        ]

    def __call__(self, ledger: MeasurementLedger) -> RebuiltCorrespondence:
        self.calls += 1
        frames = [
            frame
            for frame in ledger.frames()
            if self.camera_id is None or frame.identity.camera_id == self.camera_id
        ]
        if not frames:
            raise CameraCorrespondenceError("the ledger view holds no retained frame")
        cameras = {frame.identity.camera_id for frame in frames}
        if len(cameras) > 1:
            raise CameraCorrespondenceError(
                f"one correspondence builder covers one camera; declare camera_id among {cameras}"
            )
        frames.sort(key=lambda frame: frame.arrival_index)
        raw_frames = {
            int(frame.identity.frame_id): raw_frame_from_ledger(
                frame,
                cfa_pattern=self.cfa_pattern,
                channel_phase=self.channel_phase,
                sensor_origin_xy=self.sensor_origin_xy,
            )
            for frame in frames
            if isinstance(frame.identity.frame_id, int)
        }
        if len(raw_frames) != len(frames):
            raise CameraCorrespondenceError("every retained frame needs an integer exposure id")
        reference_id = (
            int(self.reference_frame_id)
            if self.reference_frame_id is not None
            else int(frames[-1].identity.frame_id)
        )
        if reference_id not in raw_frames:
            raise CameraCorrespondenceError(
                f"reference frame {reference_id} is not in the retained view"
            )
        reference = raw_frames[reference_id]
        maps: list[RawCorrespondenceMap] = []
        for frame in frames:
            frame_id = int(frame.identity.frame_id)
            if frame_id == reference_id:
                continue
            maps.append(
                self._estimator.estimate(
                    reference, raw_frames[frame_id], iters=self._iters, taps=True
                )
            )

        columns = self.usable_columns(ledger)
        column_index = {key.identity(): position for position, key in enumerate(columns)}
        channels, height, width = reference.channels, *reference.packed_shape
        row_positions: list[tuple[float, float]] = []
        rows: list[torch.Tensor] = []
        row_risk: list[float] = []
        kinds: list[str] = []
        versions: list[str] = []
        dropped = 0
        risk_available = any(estimated.flow_risk is not None for estimated in maps)
        for channel in range(channels):
            for row in range(height):
                for col in range(width):
                    key = reference.sample_key(channel, row, col)
                    if not ledger.has_source(key) or ledger.validity_of(key) != "valid":
                        continue
                    weights: dict[tuple[Any, ...], float] = {}
                    risk_sum = 0.0
                    total = 0.0
                    for estimated in maps:
                        kinds.append(estimated.risk_kind)
                        versions.append(estimated.mapping_version)
                        if estimated.source_identity is None:
                            continue
                        validity = estimated.valid
                        if validity is False or not bool(validity[channel, row, col]):
                            continue
                        for tap, weight in estimated.taps_for_query(channel, row, col):
                            if weight == 0.0:
                                continue
                            if not ledger.has_source(tap) or ledger.validity_of(tap) != "valid":
                                raise CameraCorrespondenceError(
                                    "a tap names a source that is not usable in this view: "
                                    f"{tap.identity()}"
                                )
                            weights[tap.identity()] = weights.get(tap.identity(), 0.0) + float(
                                weight
                            )
                            total += float(weight)
                            if estimated.flow_risk is not None:
                                risk_sum += float(weight) * float(estimated.flow_risk[row, col])
                    if total <= 0.0:
                        dropped += 1
                        continue
                    vector = torch.zeros(len(columns), dtype=torch.float64)
                    for identity, weight in weights.items():
                        position = column_index.get(identity)
                        if position is None:
                            raise CameraCorrespondenceError(
                                "a correspondence tap references a source outside the declared "
                                f"evidence set: {identity}"
                            )
                        vector[position] = weight / total
                    row_positions.append(ledger.sample_coordinates(key))
                    rows.append(vector)
                    row_risk.append(risk_sum / total)
        if not rows:
            raise CameraCorrespondenceError(
                "the retained view produced no valid reference query with a tap"
            )
        weights_tensor = torch.stack(rows, dim=0)
        risk = torch.tensor(row_risk, dtype=torch.float64) if risk_available else None
        return RebuiltCorrespondence(
            row_positions=tuple(row_positions),
            weights=weights_tensor,
            version=f"correspondence-{_uniform(versions, 'n01-full-v1')}",
            source_ids=tuple(columns),
            mapping_version=_uniform(versions, "n01-full-v1"),
            decision_version=self.decision_version,
            risk_kind=_risk_kind(kinds),
            reference_frame_id=reference_id,
            source_frame_ids=tuple(int(frame.identity.frame_id) for frame in frames),
            valid_rows=len(rows),
            dropped_rows=dropped,
            risk=risk,
        )


@dataclass(frozen=True)
class LedgerNoiseModelBuilder:
    """Recompute the P2 noise model from the retained RAW means of one view.

    ``source_ids`` declares the identities to model (the domain's evidence set);
    absent or non-valid samples in the passed view are skipped, exactly like
    ``fit_structure`` does.  Means and block keys are read from the view, and the
    shot/read law itself comes from a versioned
    :class:`~owac.raw.n01_noise.NoiseCalibration`, so no value is cached.
    """

    calibration: NoiseCalibration
    source_ids: tuple[Any, ...]
    correlation: str = "declared"

    @property
    def version(self) -> str:
        return self.calibration.noise_version

    def __call__(self, ledger: MeasurementLedger) -> NoiseModel:
        if not self.source_ids:
            raise ValueError("LedgerNoiseModelBuilder.source_ids must be non-empty")
        order = []
        means = []
        for key in self.source_ids:
            if not ledger.has_source(key) or ledger.validity_of(key) != "valid":
                continue
            order.append(key)
            means.append(float(ledger.get_source(key).raw_value))
        if not order:
            raise ValueError("the view holds no usable source for the noise model")
        return build_noise_model(
            order,
            means,
            self.calibration,
            correlation=self.correlation,
            version=self.calibration.noise_version,
        )

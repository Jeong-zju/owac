"""RAW-domain correspondence adapter over the pinned SEA-RAFT motion module.

The estimator reuses the author feature extractor, correlation pyramid and
iterative update (vendored at a fixed commit under
``src/owac/baselines/_vendor/sea_raft``) and adds only the RAW input contract,
the coordinate/unit conversion and the sampling coefficients.  The prompt of
record is ``docs/prompts/n01_v1_0/P2_RAW_MOTION.md`` sections 2 and 3.

Geometry contract
-----------------
* :class:`RawSensorFrame` holds one arrived exposure with ``values`` shaped
  ``[C, Hp, Wp]``.  ``cfa4`` packs the four CFA phases into four
  half-resolution planes (``Hp = H / 2``, ``Wp = W / 2``), so **one packed cell
  step is two sensor pixels**; ``rgb3`` keeps three co-located full-resolution
  channels, so one packed cell step is one sensor pixel.
* ``sensor_origin_xy`` is the full-sensor ``(y, x)`` of ``values[0, 0, 0]``.
  Channel ``k`` carries phase ``p_k = channel_phase[k]`` with parity offset
  ``off(p) = (p // 2, p % 2)`` and sits at sensor
  ``(y0 + 2 i + d_y(k), x0 + 2 j + d_x(k))`` with
  ``d(k) = off(p_k) - off(p_0)``.  The four phases therefore share one packed
  index space but each has its own sensor offset, and a declared frame is
  refused when its origin parity contradicts ``channel_phase[0]``.
* :meth:`CorrespondenceEstimator.estimate` returns a **pull** map:
  ``source_xy[k, i, j]`` is the continuous source-frame *sensor* coordinate to
  sample when the reference wants its own sample ``(k, i, j)``.  It is never a
  push map and never a packed index.
* Sampling is bilinear into the source frame.  The four taps are real source
  samples of the same phase lattice (CFA) or of each co-located channel
  (rgb3), each named by a :class:`SourceKey` whose ``sensor_y``/``sensor_x`` and
  ``channel_or_phase`` match the source frame's actual samples.  Out-of-frame
  taps are never wrapped and never filled with a fabricated sample: a query
  whose contributing taps leave the source frame is invalid.
* The estimator is memoryless.  It only ever sees the two frames passed to it,
  so no call can read a future frame; forward/backward consistency comes from
  :meth:`CorrespondenceEstimator.estimate_bidirectional`, which runs the same
  already-arrived pair twice.
* ``visibility`` and ``flow_risk`` are **proxies** derived from the author
  mixture head.  They are not a posterior covariance and not a calibrated
  probability.  :class:`RiskCalibrationTable` is the declared protocol for an
  independent calibration table to map them; until one is supplied the result
  reports ``risk_kind == "proxy"``.

Value contract
--------------
The vendored author code no longer rescales its input with ``2 * (x / 255) - 1``.
RAW values reach the network exactly as supplied, so any normalisation must be
an explicit :class:`RawNormalization` (declared sensor/exposure calibration or a
declared fixed interval).  Per-image min/max and RGB-style mapping are not
representable and are refused.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import torch

from owac.baselines._vendor.sea_raft.raft import RAFT
from owac.baselines._vendor.sea_raft.utils.utils import bilinear_sampler
from owac.frameworks.n01_full.config import CFA_PATTERNS, CorrespondenceConfig
from owac.frameworks.n01_full.contracts import CorrespondenceMap, SourceKey

__all__ = [
    "MODES",
    "MODE_CHANNELS",
    "SENSOR_PER_PACKED",
    "CFA_PHASE_OFFSETS",
    "BidirectionalCorrespondence",
    "CorrespondenceEstimator",
    "PackedFlowModel",
    "RawCorrespondenceMap",
    "RawNormalization",
    "RawSensorFrame",
    "RiskCalibrationTable",
]

#: The only two RAW stems this adapter builds: four CFA phases packed as four
#: planes (fnet 4 / cnet 8) and three co-located pseudo-RAW channels
#: (fnet 3 / cnet 6).  No third mode exists.
MODES = ("cfa4", "rgb3")

MODE_CHANNELS = {"cfa4": 4, "rgb3": 3}

#: Sensor pixels per packed cell step, per axis.
SENSOR_PER_PACKED = {"cfa4": 2, "rgb3": 1}

#: ``off(p) = (dy, dx)`` of CFA phase ``p`` in sensor pixels.
CFA_PHASE_OFFSETS = tuple((p // 2, p % 2) for p in range(4))

_DEFAULT_MAPPING_VERSION = "n01-full-v1"
_UNVERSIONED = "unversioned"


def _phase_of_sensor(y: int, x: int) -> int:
    """CFA phase index (0..3) of a full-sensor coordinate; complement of off()."""
    return 2 * (y % 2) + (x % 2)


def _source_key(
    identity: tuple[str, str, str],
    frame_id: int | str,
    channel_or_phase: int | str,
    sensor_y: int,
    sensor_x: int,
) -> SourceKey:
    """One original source sample; the only place a tap identity is built."""
    dataset_id, episode_id, camera_id = identity
    return SourceKey(
        dataset_id=dataset_id,
        episode_id=episode_id,
        camera_id=camera_id,
        frame_id=frame_id,
        channel_or_phase=channel_or_phase,
        sensor_y=sensor_y,
        sensor_x=sensor_x,
    )


def _packed_pull_geometry(
    packed_flow: torch.Tensor,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]
]:
    """Four bilinear pull taps of a packed displacement field.

    ``packed_flow`` is ``[2, Hp, Wp]`` in packed cells.  Returns
    ``(finite, tap_rows, tap_cols, tap_weights)`` where each tap tuple has the
    four corners in the order ``(i0,j0), (i0,j1), (i1,j0), (i1,j1)`` and
    the weights are ``(1-fy)(1-fx), (1-fy)fx, fy(1-fx), fy fx``.
    """
    device = packed_flow.device
    rows = torch.arange(packed_flow.shape[1], device=device, dtype=torch.float64).view(-1, 1)
    cols = torch.arange(packed_flow.shape[2], device=device, dtype=torch.float64).view(1, -1)
    u = rows + packed_flow[0].to(torch.float64)
    v = cols + packed_flow[1].to(torch.float64)
    finite = torch.isfinite(u) & torch.isfinite(v)
    safe_u = torch.where(finite, u, torch.zeros_like(u))
    safe_v = torch.where(finite, v, torch.zeros_like(v))
    i0 = torch.floor(safe_u)
    j0 = torch.floor(safe_v)
    fy = safe_u - i0
    fx = safe_v - j0
    tap_rows = (i0, i0, i0 + 1, i0 + 1)
    tap_cols = (j0, j0 + 1, j0, j0 + 1)
    tap_weights = ((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx)
    return finite, tap_rows, tap_cols, tap_weights


@dataclass(frozen=True)
class RawSensorFrame:
    """One arrived RAW sample; coordinates are always full-sensor ``xy``.

    ``values`` is ``[C, Hp, Wp]`` in RAW or calibrated units, ``float32``:
    ``C = 4`` half-resolution CFA phase planes (``cfa4``) or ``C = 3``
    co-located pseudo-RAW channels (``rgb3``).  ``channel_phase`` maps the
    packed channel index to its CFA phase (a permutation of ``0..3``) and is
    ``()`` for ``rgb3``.  ``sensor_origin_xy`` is the full-sensor ``(y, x)`` of
    ``values[0, 0, 0]``; channel ``k``'s packed ``(i, j)`` sample sits at
    ``sensor_origin_xy + (2 i, 2 j) + d(k)`` with
    ``d(k) = off(channel_phase[k]) - off(channel_phase[0])`` for CFA and the
    identity for ``rgb3``.

    ``validity``, when given, is a ``[C, Hp, Wp]`` bool mask of usable samples
    (``None`` means every sample is usable); missing, bad and saturated samples
    are excluded from taps rather than replaced by zeros.  ``frame_id`` is the
    exposure identity, so a repeated transmission of one frame carries one key
    while two exposures with equal values stay distinct.

    Re-decoding a different packing (for example a fourth fabricated phase for
    ``rgb3``) is not representable: the constructor refuses a ``cfa_pattern``,
    channel count and ``channel_phase`` that disagree.
    """

    dataset_id: str
    episode_id: str | int
    camera_id: str | int
    frame_id: int
    values: torch.Tensor
    channel_phase: tuple[int, ...]
    cfa_pattern: str
    sensor_origin_xy: tuple[int, int]
    validity: torch.Tensor | None = None
    time: float = 0.0
    calibration_version: str = _UNVERSIONED

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("RawSensorFrame.dataset_id must be a non-empty identity string")
        for name in ("episode_id", "camera_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, str | int) or value == "":
                raise ValueError(f"RawSensorFrame.{name} must be a non-empty str or int identity")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise TypeError("RawSensorFrame.frame_id must be the integer exposure identity")
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 3:
            raise TypeError("RawSensorFrame.values must be a [C, Hp, Wp] torch.Tensor")
        if not self.values.dtype.is_floating_point:
            raise TypeError(
                "RawSensorFrame.values must be a floating RAW/calibrated tensor; "
                "quantised integers belong in the measurement ledger"
            )
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("RawSensorFrame.values must be finite; a gap is not a value")
        if not isinstance(self.channel_phase, tuple) or not all(
            isinstance(p, int) and not isinstance(p, bool) for p in self.channel_phase
        ):
            raise TypeError("RawSensorFrame.channel_phase must be a tuple of phase indices")
        if not isinstance(self.sensor_origin_xy, tuple) or len(self.sensor_origin_xy) != 2:
            raise TypeError("RawSensorFrame.sensor_origin_xy must be an (y, x) tuple")
        for value in self.sensor_origin_xy:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("RawSensorFrame.sensor_origin_xy must hold non-negative ints")
        if self.validity is not None:
            if not isinstance(self.validity, torch.Tensor) or self.validity.dtype != torch.bool:
                raise TypeError("RawSensorFrame.validity must be a bool tensor or None")
            if tuple(self.validity.shape) != tuple(self.values.shape):
                raise ValueError(
                    "RawSensorFrame.validity must have the same [C, Hp, Wp] shape as values"
                )
        if not isinstance(self.calibration_version, str) or not self.calibration_version:
            raise ValueError("RawSensorFrame.calibration_version must be declared")
        if not self.dataset_id or not self.calibration_version:
            raise ValueError("RawSensorFrame identity and calibration must be declared")

        channels = self.values.shape[0]
        if self.cfa_pattern in CFA_PATTERNS:
            if channels != 4:
                raise ValueError(
                    f"CFA pattern {self.cfa_pattern!r} carries four packed phases, "
                    f"got C={channels}; a fabricated fourth measurement is refused"
                )
            if sorted(self.channel_phase) != [0, 1, 2, 3]:
                raise ValueError(
                    f"channel_phase {self.channel_phase!r} must be a permutation of (0, 1, 2, 3)"
                )
            y0, x0 = self.sensor_origin_xy
            if _phase_of_sensor(y0, x0) != self.channel_phase[0]:
                raise ValueError(
                    f"sensor origin {(y0, x0)} has CFA phase {_phase_of_sensor(y0, x0)}, but "
                    f"channel 0 carries phase {self.channel_phase[0]}; the declared origin and "
                    "packing disagree"
                )
        elif self.cfa_pattern == "rgb3":
            if channels != 3:
                raise ValueError("rgb3 carries three co-located channels, got C=" + str(channels))
            if self.channel_phase != ():
                raise ValueError(
                    f"rgb3 has no CFA phase, got channel_phase={self.channel_phase!r}; do not "
                    "fabricate a fourth measurement or a CFA phase"
                )
        else:
            raise ValueError(
                f"Unknown cfa_pattern {self.cfa_pattern!r}; expected one of "
                f"{CFA_PATTERNS} or 'rgb3'"
            )
        if self.cfa_pattern == "rgb3" and self.values.shape[0] != 3:
            raise ValueError("rgb3 requires C=3")

    @property
    def mode(self) -> str:
        """``"cfa4"`` or ``"rgb3"``; the stem this frame feeds."""
        return "cfa4" if self.cfa_pattern in CFA_PATTERNS else "rgb3"

    @property
    def channels(self) -> int:
        """Number of packed channels."""
        return int(self.values.shape[0])

    @property
    def packed_shape(self) -> tuple[int, int]:
        """``(Hp, Wp)`` of the packed grid."""
        return int(self.values.shape[1]), int(self.values.shape[2])

    @property
    def sensor_per_packed(self) -> int:
        """Sensor pixels per packed cell step (2 for CFA, 1 for rgb3)."""
        return SENSOR_PER_PACKED[self.mode]

    @property
    def sensor_shape(self) -> tuple[int, int]:
        """Sensor footprint spanned by the packed grid, in pixels."""
        hp, wp = self.packed_shape
        step = self.sensor_per_packed
        return step * hp, step * wp

    def phase_of_channel(self, channel: int) -> int | None:
        """CFA phase carried by ``channel``, or ``None`` for rgb3."""
        if self.mode == "rgb3":
            return None
        return self.channel_phase[channel]

    def phase_offset(self, channel: int) -> tuple[int, int]:
        """Sensor offset of ``channel`` relative to channel 0's lattice."""
        if self.mode == "rgb3":
            return (0, 0)
        phase = self.channel_phase[channel]
        anchor = self.channel_phase[0]
        return (
            CFA_PHASE_OFFSETS[phase][0] - CFA_PHASE_OFFSETS[anchor][0],
            CFA_PHASE_OFFSETS[phase][1] - CFA_PHASE_OFFSETS[anchor][1],
        )

    def sensor_xy_of(self, channel: int, row: int, col: int) -> tuple[int, int]:
        """Full-sensor ``(y, x)`` of the packed sample ``(channel, row, col)``."""
        step = self.sensor_per_packed
        dy, dx = self.phase_offset(channel)
        y0, x0 = self.sensor_origin_xy
        return y0 + step * row + dy, x0 + step * col + dx

    def valid_at(self, channel: int, row: int, col: int) -> bool:
        """Whether the source sample ``(channel, row, col)`` is usable."""
        if self.validity is None:
            return True
        return bool(self.validity[channel, row, col])

    def sample_key(self, channel: int, row: int, col: int) -> SourceKey:
        """Identity of this frame's own sample at ``(channel, row, col)``."""
        hp, wp = self.packed_shape
        if not (0 <= channel < self.channels and 0 <= row < hp and 0 <= col < wp):
            raise IndexError(f"packed sample ({channel}, {row}, {col}) is outside this frame")
        phase = self.phase_of_channel(channel)
        channel_or_phase: int | str = channel if phase is None else phase
        y, x = self.sensor_xy_of(channel, row, col)
        return _source_key(self.identity, self.frame_id, channel_or_phase, y, x)

    @property
    def identity(self) -> tuple[str, str, str]:
        """``(dataset_id, episode_id, camera_id)`` as identity strings."""
        return self.dataset_id, str(self.episode_id), str(self.camera_id)


@dataclass(frozen=True)
class RawNormalization:
    """Declared RAW -> network-input value transform ``(x - center) / half_range``.

    ``method`` is ``"affine_calibration"`` (black/white level, exposure or gain
    calibration with a named version) or ``"fixed_interval"`` (a declared fixed
    interval, not a per-image statistic).  Per-image min/max cannot be expressed
    here: the spec is a value object built before any frame is seen, and the
    estimator refuses a spec whose ``source`` does not name the calibration
    artifact it came from.
    """

    method: str
    center: float
    half_range: float
    source: str
    calibration_version: str = _UNVERSIONED

    def __post_init__(self) -> None:
        if self.method not in ("affine_calibration", "fixed_interval"):
            raise ValueError(
                f"Unknown RawNormalization.method {self.method!r}; expected "
                "'affine_calibration' or 'fixed_interval'"
            )
        if not math.isfinite(self.center):
            raise ValueError("RawNormalization.center must be finite")
        if not math.isfinite(self.half_range) or self.half_range <= 0:
            raise ValueError("RawNormalization.half_range must be finite and positive")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(
                "RawNormalization.source must name the calibration artifact or declared interval"
            )

    @classmethod
    def fixed_interval(
        cls,
        low: float,
        high: float,
        *,
        source: str,
        calibration_version: str = _UNVERSIONED,
    ) -> RawNormalization:
        """Map a declared fixed interval ``[low, high]`` onto ``[-1, 1]``."""
        if not (math.isfinite(low) and math.isfinite(high) and high > low):
            raise ValueError("a fixed interval needs finite low < high")
        return cls(
            method="fixed_interval",
            center=(low + high) / 2.0,
            half_range=(high - low) / 2.0,
            source=source,
            calibration_version=calibration_version,
        )

    @classmethod
    def affine_calibration(
        cls,
        black_level: float,
        white_level: float,
        *,
        source: str,
        calibration_version: str = _UNVERSIONED,
    ) -> RawNormalization:
        """Map calibrated RAW counts in ``(black_level, white_level]`` to ``[-1, 1]``."""
        if not (math.isfinite(black_level) and math.isfinite(white_level)):
            raise ValueError("black/white level must be finite")
        if white_level <= black_level:
            raise ValueError("white_level must exceed black_level")
        return cls(
            method="affine_calibration",
            center=(black_level + white_level) / 2.0,
            half_range=(white_level - black_level) / 2.0,
            source=source,
            calibration_version=calibration_version,
        )

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Value transform; no per-image statistic is computed here."""
        return (values - self.center) / self.half_range


@runtime_checkable
class RiskCalibrationTable(Protocol):
    """Protocol for the independent proxy -> calibrated-risk table (P2 owner:
    motion-train-eval).  ``map_proxy`` receives the packed-grid proxy scores and
    returns ``(calibrated_risk, coverage)`` with the same ``[Hp, Wp]`` shape.
    """

    version: str

    def map_proxy(
        self, *, flow_risk: torch.Tensor, visibility: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class PackedFlowModel(Protocol):
    """Structural type of the motion model: the vendored RAFT call signature."""

    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int | None = None,
        flow_gt: torch.Tensor | None = None,
        test_mode: bool = False,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RawCorrespondenceMap(CorrespondenceMap):
    """A :class:`CorrespondenceMap` that carries its RAW geometry explicitly.

    Queries are the measured reference samples, indexed ``[channel, row, col]``
    on the packed grid, so ``source_xy`` is ``[C, Hp, Wp, 2]`` and ``valid`` is
    ``[C, Hp, Wp]``.  ``tap_source_ids`` / ``tap_weights`` are flattened in
    ``(channel, row, col, tap)`` order over valid queries only, each query
    contributing its nonzero-weight taps in the order
    ``(i0,j0), (i0,j1), (i1,j0), (i1,j1)``.

    ``visibility`` and ``flow_risk`` are packed-grid ``[Hp, Wp]`` **proxies**
    from the author mixture head; ``risk_kind`` is ``"proxy"`` until an
    independent :class:`RiskCalibrationTable` supplies ``calibrated_risk`` and
    ``coverage`` (then it is ``"calibrated:<table version>"``).  Comparing two
    maps with ``==`` is ambiguous because the fields are tensors; compare the
    fields explicitly.

    ``packed_flow`` is the raw ``[2, Hp, Wp]`` packed displacement field the
    network produced (one packed cell = ``sensor_per_packed`` sensor pixels),
    kept so the geometry can be re-derived without re-running the network.
    """

    mode: str = "rgb3"
    packed_flow: Any = None
    channel_phase: tuple[int, ...] = ()
    sensor_per_packed: int = 1
    reference_sensor_origin_xy: tuple[int, int] = (0, 0)
    source_sensor_origin_xy: tuple[int, int] = (0, 0)
    source_identity: tuple[str, str, str] | None = None
    risk_kind: str = "unavailable"
    normalization_source: str = "caller_pre_normalized"
    calibrated_risk: Any = None
    coverage: Any = None

    @property
    def packed_shape(self) -> tuple[int, int]:
        """``(Hp, Wp)`` of the packed grid this map was estimated on."""
        return int(self.packed_flow.shape[1]), int(self.packed_flow.shape[2])

    def phase_of_channel(self, channel: int) -> int | None:
        """CFA phase of ``channel``, or ``None`` for rgb3."""
        if self.mode == "rgb3":
            return None
        return self.channel_phase[channel]

    def phase_offset(self, channel: int) -> tuple[int, int]:
        """Sensor offset of ``channel`` relative to channel 0's lattice."""
        if self.mode == "rgb3":
            return (0, 0)
        phase = self.channel_phase[channel]
        anchor = self.channel_phase[0]
        return (
            CFA_PHASE_OFFSETS[phase][0] - CFA_PHASE_OFFSETS[anchor][0],
            CFA_PHASE_OFFSETS[phase][1] - CFA_PHASE_OFFSETS[anchor][1],
        )

    def reference_sensor_xy(self, channel: int, row: int, col: int) -> tuple[int, int]:
        """Full-sensor ``(y, x)`` of the reference sample that query names."""
        step = self.sensor_per_packed
        dy, dx = self.phase_offset(channel)
        y0, x0 = self.reference_sensor_origin_xy
        return y0 + step * row + dy, x0 + step * col + dx

    def taps_for_query(
        self, channel: int, row: int, col: int
    ) -> tuple[tuple[SourceKey, float], ...]:
        """Re-derive one query's nonzero bilinear taps from ``packed_flow``."""
        if self.source_identity is None:
            raise ValueError(
                "this map does not carry the source identity needed to rebuild tap keys"
            )
        hp, wp = self.packed_shape
        if not (0 <= channel < MODE_CHANNELS[self.mode] and 0 <= row < hp and 0 <= col < wp):
            raise IndexError(f"query ({channel}, {row}, {col}) is outside the packed grid")
        finite, tap_rows, tap_cols, tap_weights = _packed_pull_geometry(self.packed_flow)
        phase = self.phase_of_channel(channel)
        channel_or_phase: int | str = channel if phase is None else phase
        step = self.sensor_per_packed
        dy, dx = self.phase_offset(channel)
        y0, x0 = self.source_sensor_origin_xy
        taps: list[tuple[SourceKey, float]] = []
        for tap in range(4):
            weight = float(tap_weights[tap][row, col])
            if weight == 0.0 or not bool(finite[row, col]):
                continue
            i = int(tap_rows[tap][row, col].item())
            j = int(tap_cols[tap][row, col].item())
            if not (0 <= i < hp and 0 <= j < wp):
                continue
            taps.append(
                (
                    _source_key(
                        self.source_identity,
                        self.source_frame_id,
                        channel_or_phase,
                        y0 + step * i + dy,
                        x0 + step * j + dx,
                    ),
                    weight,
                )
            )
        return tuple(taps)


@dataclass(frozen=True)
class BidirectionalCorrespondence:
    """Forward/backward pull maps of one already-arrived frame pair.

    ``cycle_error`` is ``|f_forward(x) + f_backward(x + f_forward(x))|`` in
    packed cells on the reference packed grid, with the backward field sampled
    bilinearly at the forward-mapped point.  ``consistent`` marks the packed
    cells whose round trip stays inside the frame and within ``tolerance``;
    per-query ``valid`` masks stay on the two maps and are not silently folded
    into this diagnostic.
    """

    forward: RawCorrespondenceMap
    backward: RawCorrespondenceMap
    cycle_error: torch.Tensor
    consistent: torch.Tensor
    tolerance: float


class CorrespondenceEstimator:
    """RAW adapter around the pinned SEA-RAFT motion network.

    ``mode`` selects the stem: ``cfa4`` builds fnet 4 / cnet 8, ``rgb3`` builds
    fnet 3 / cnet 6.  ``depth`` (model width) and ``pretrained_checkpoint``
    (explicit weight file) are independent: the depth choice never loads
    weights, and no code path downloads torchvision ImageNet parameters.
    ``model`` injects an alternative packed-flow model (the vendored RAFT call
    signature) for tests and diagnostics.

    ``normalization`` is a :class:`RawNormalization`, a mapping keyed by
    ``frame.calibration_version``, or ``None`` when the caller already supplies
    network-range values.  ``risk_calibration`` is an optional
    :class:`RiskCalibrationTable`; without it the map's risk fields stay
    labelled proxies.
    """

    def __init__(
        self,
        *,
        mode: str,
        config: CorrespondenceConfig | None = None,
        model: torch.nn.Module | None = None,
        iters: int | None = None,
        device: str = "cpu",
        depth: str = "resnet34",
        pretrained_checkpoint: str | Path | None = None,
        normalization: RawNormalization | Mapping[str, RawNormalization] | None = None,
        risk_calibration: RiskCalibrationTable | None = None,
        var_min: float = 0.0,
        var_max: float = 10.0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        self._mode = mode
        self._channels = MODE_CHANNELS[mode]
        self._mapping_version = _DEFAULT_MAPPING_VERSION
        if config is not None:
            if not isinstance(config, CorrespondenceConfig):
                raise TypeError("config must be a CorrespondenceConfig or None")
            config.validate()
            if config.mapping_kind != "sea_raft":
                raise ValueError(
                    f"CorrespondenceConfig.mapping_kind must be 'sea_raft', got "
                    f"{config.mapping_kind!r}"
                )
            if config.input_channels is not None and config.input_channels not in (
                self._channels,
                2 * self._channels,
            ):
                raise ValueError(
                    f"declared input_channels {config.input_channels} is not the {mode} stem "
                    f"({self._channels} or {2 * self._channels})"
                )
            if mode == "cfa4" and not config.preserve_cfa_phase:
                raise ValueError("cfa4 keeps the sampled CFA phase; preserve_cfa_phase is required")
            self._mapping_version = config.mapping_version
        if isinstance(iters, bool) or (
            iters is not None and (not isinstance(iters, int) or iters < 0)
        ):
            raise ValueError("iters must be a non-negative int or None")
        if depth not in ("resnet18", "resnet34"):
            raise ValueError(f"Unknown depth {depth!r}; expected 'resnet18' or 'resnet34'")
        if not (math.isfinite(var_min) and math.isfinite(var_max) and var_min <= 0.0 <= var_max):
            raise ValueError(
                "the author mixture head clamps its two log-scales to [var_min, 0] and "
                "[0, var_max]; var_min <= 0 <= var_max is required"
            )
        if normalization is not None and not isinstance(normalization, RawNormalization | Mapping):
            raise TypeError(
                "normalization must be a RawNormalization, a mapping keyed by "
                "calibration_version, or None"
            )
        if isinstance(normalization, Mapping):
            for key, spec in normalization.items():
                if not isinstance(key, str) or not isinstance(spec, RawNormalization):
                    raise TypeError(
                        "a normalization mapping must be keyed by calibration_version strings "
                        "and hold RawNormalization values"
                    )
        if risk_calibration is not None and not isinstance(risk_calibration, RiskCalibrationTable):
            raise TypeError(
                "risk_calibration must implement RiskCalibrationTable (version + map_proxy)"
            )

        self._device = torch.device(device)
        self._iters = 4 if iters is None else iters
        self._depth = depth
        self._normalization = normalization
        self._risk_calibration = risk_calibration
        self._var_min = float(var_min)
        self._var_max = float(var_max)
        self._pretrained_report: Mapping[str, Any] | None = None

        if model is None:
            args = _SeaRaftArgs(iters=self._iters)
            self._model = RAFT(args, input_dim=self._channels, depth=depth).to(self._device).eval()
            self._owns_model = True
        else:
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model must be a torch.nn.Module implementing the RAFT call")
            self._model = model.to(self._device).eval()
            self._owns_model = False
        if pretrained_checkpoint is not None:
            self._pretrained_report = self._load_pretrained(pretrained_checkpoint)

    # -- model surface ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """The RAW stem this estimator builds."""
        return self._mode

    @property
    def depth(self) -> str:
        """Model-depth choice; independent of any weight file."""
        return self._depth

    @property
    def iters(self) -> int:
        """Default number of iterative update steps."""
        return self._iters

    @property
    def device(self) -> torch.device:
        """Device the model and inputs live on."""
        return self._device

    @property
    def model(self) -> torch.nn.Module:
        """The underlying packed-flow model (the trainer's entry point)."""
        return self._model

    @property
    def mapping_version(self) -> str:
        """Version string recorded on every emitted map."""
        return self._mapping_version

    @property
    def pretrained_report(self) -> Mapping[str, Any] | None:
        """Which tensors an explicit checkpoint loaded; ``None`` when unused."""
        if self._pretrained_report is None:
            return None
        return MappingProxyType(dict(self._pretrained_report))

    def model_parameters(self):
        """Trainable parameters of the motion model, for the trainer."""
        return self._model.parameters()

    def preprocess(self, frame: RawSensorFrame) -> torch.Tensor:
        """Normalize one frame into the ``[1, C, Hp, Wp]`` network input."""
        self._check_frame(frame, "frame")
        spec = self._spec_for(frame)
        values = frame.values.to(device=self._device, dtype=torch.float32)
        if spec is not None:
            values = spec.apply(values)
        return values.unsqueeze(0)

    def _load_pretrained(self, checkpoint: str | Path) -> Mapping[str, Any]:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, Mapping):
            raise TypeError(f"checkpoint {path} does not hold a tensor state dict")
        report = self._model.load_state_dict(state, strict=False)
        loaded = len(state) - len(report.unexpected_keys)
        if loaded <= 0:
            raise RuntimeError(
                f"checkpoint {path} loaded no tensor into the {self._mode} stem; the file does "
                "not belong to this model"
            )
        return {
            "path": str(path),
            "loaded_tensors": int(loaded),
            "missing_keys": tuple(report.missing_keys),
            "unexpected_keys": tuple(report.unexpected_keys),
        }

    # -- estimation ---------------------------------------------------------------------

    def estimate(
        self,
        reference: RawSensorFrame,
        source: RawSensorFrame,
        *,
        iters: int | None = None,
        taps: bool = True,
    ) -> RawCorrespondenceMap:
        """Estimate the pull map from the current reference frame to one past source.

        ``source_xy[k, i, j]`` is the continuous *source-frame sensor* coordinate
        (pull) that the reference sample ``(k, i, j)`` reads; a positive
        displacement therefore points from the reference to the source, never the
        other way.  ``source_xy`` at an invalid query is not meaningful (a
        non-finite network flow stays non-finite there); read ``valid`` first.
        ``taps=False`` skips the dense coefficient table (large frames); the
        coefficients stay re-derivable through
        :meth:`RawCorrespondenceMap.taps_for_query`.
        """
        self._check_pair(reference, source)
        effective_iters = self._effective_iters(iters)
        hp, wp = reference.packed_shape
        if self._owns_model and effective_iters > 0:
            # The author correlation pyramid halves the 1/8 feature map once per
            # level, so the last level needs a feature map of at least 2**levels.
            needed = 8 * 2**_SeaRaftArgs.CORR_LEVELS
            if (
                -(-hp // 8) < 2**_SeaRaftArgs.CORR_LEVELS
                or -(-wp // 8) < 2**_SeaRaftArgs.CORR_LEVELS
            ):
                raise ValueError(
                    f"the vendored {_SeaRaftArgs.CORR_LEVELS}-level correlation pyramid needs "
                    f"at least {needed} packed cells per side, got {(hp, wp)}"
                )
        reference_input = self.preprocess(reference)
        source_input = self.preprocess(source)
        with torch.no_grad(), self._eval_mode():
            outputs = self._model(
                reference_input, source_input, iters=effective_iters, test_mode=True
            )
        flow = self._extract_flow(outputs, reference.packed_shape)
        proxy = self._proxy_scores(outputs, flow.shape[1:])

        channels = reference.channels
        finite, tap_rows, tap_cols, tap_weights = _packed_pull_geometry(flow)
        contributing = tuple(weight > 0.0 for weight in tap_weights)
        in_frame = tuple(
            ((rows >= 0) & (rows < hp) & (cols >= 0) & (cols < wp)) | ~contributes
            for rows, cols, contributes in zip(tap_rows, tap_cols, contributing, strict=True)
        )
        spatial_ok = torch.stack(in_frame).all(dim=0) & finite

        validity = source.validity
        if validity is None:
            valid = spatial_ok.unsqueeze(0).expand(channels, hp, wp).clone()
        else:
            per_tap = []
            for rows, cols, contributes in zip(tap_rows, tap_cols, contributing, strict=True):
                ii = rows.clamp(0, hp - 1).long()
                jj = cols.clamp(0, wp - 1).long()
                per_tap.append(validity[:, ii, jj] | ~contributes)
            valid = torch.stack(per_tap).all(dim=0) & spatial_ok.unsqueeze(0)

        origins = torch.tensor(
            [source.phase_offset(channel) for channel in range(channels)],
            dtype=torch.float64,
            device=self._device,
        )
        step = source.sensor_per_packed
        y0, x0 = source.sensor_origin_xy
        pull_y = (
            y0
            + step
            * (
                torch.arange(hp, device=self._device, dtype=torch.float64).view(-1, 1)
                + flow[0].to(torch.float64)
            )
        ).unsqueeze(0) + origins[:, 0].view(-1, 1, 1)
        pull_x = (
            x0
            + step
            * (
                torch.arange(wp, device=self._device, dtype=torch.float64).view(1, -1)
                + flow[1].to(torch.float64)
            )
        ).unsqueeze(0) + origins[:, 1].view(-1, 1, 1)
        source_xy = torch.stack([pull_y, pull_x], dim=-1)

        whole_rejected = not bool(valid.any())
        tap_ids: tuple[SourceKey, ...] = ()
        tap_values: tuple[float, ...] = ()
        if taps and not whole_rejected:
            tap_ids, tap_values = self._build_taps(source, valid, tap_rows, tap_cols, tap_weights)

        risk_kind, calibrated_risk, coverage = self._calibrate(proxy)
        return RawCorrespondenceMap(
            source_frame_id=source.frame_id,
            reference_frame_id=reference.frame_id,
            source_xy=source_xy,
            tap_source_ids=tap_ids,
            tap_weights=tap_values,
            valid=False if whole_rejected else valid,
            mapping_version=self._mapping_version,
            visibility=None if proxy is None else proxy["visibility"],
            flow_risk=None if proxy is None else proxy["flow_risk"],
            mode=self._mode,
            packed_flow=flow,
            channel_phase=source.channel_phase,
            sensor_per_packed=step,
            reference_sensor_origin_xy=reference.sensor_origin_xy,
            source_sensor_origin_xy=source.sensor_origin_xy,
            source_identity=None if whole_rejected else source.identity,
            risk_kind=risk_kind,
            normalization_source=self._normalization_source(reference, source),
            calibrated_risk=calibrated_risk,
            coverage=coverage,
        )

    def estimate_bidirectional(
        self,
        reference: RawSensorFrame,
        source: RawSensorFrame,
        *,
        iters: int | None = None,
        tolerance: float = 1.0,
        taps: bool = False,
    ) -> BidirectionalCorrespondence:
        """Run the same arrived pair twice (forward and backward) and compare.

        No third frame is involved, so this cannot read a future exposure.
        ``tolerance`` is the cycle threshold in packed cells (default one cell).
        """
        if tolerance <= 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be a finite positive packed-cell distance")
        forward = self.estimate(reference, source, iters=iters, taps=taps)
        backward = self.estimate(source, reference, iters=iters, taps=taps)
        forward_flow = forward.packed_flow
        hp, wp = forward.packed_shape
        rows = torch.arange(hp, device=self._device, dtype=torch.float64).view(-1, 1)
        cols = torch.arange(wp, device=self._device, dtype=torch.float64).view(1, -1)
        mapped_y = rows + forward_flow[0].to(torch.float64)
        mapped_x = cols + forward_flow[1].to(torch.float64)
        coords = torch.stack([mapped_x, mapped_y], dim=-1).unsqueeze(0).to(forward_flow.dtype)
        sampled = bilinear_sampler(backward.packed_flow.unsqueeze(0), coords)[0].to(torch.float64)
        cycle = forward_flow.to(torch.float64) + sampled
        cycle_error = torch.linalg.vector_norm(cycle, dim=0)
        inside = (
            (mapped_y >= 0)
            & (mapped_y <= hp - 1)
            & (mapped_x >= 0)
            & (mapped_x <= wp - 1)
            & torch.isfinite(mapped_y)
            & torch.isfinite(mapped_x)
        )
        return BidirectionalCorrespondence(
            forward=forward,
            backward=backward,
            cycle_error=cycle_error,
            consistent=(cycle_error <= tolerance) & inside,
            tolerance=float(tolerance),
        )

    # -- internals ----------------------------------------------------------------------

    def _check_frame(self, frame: RawSensorFrame, role: str) -> None:
        if not isinstance(frame, RawSensorFrame):
            raise TypeError(f"{role} must be a RawSensorFrame, got {type(frame).__name__}")
        if frame.mode != self._mode:
            raise ValueError(
                f"{role} is a {frame.mode} frame but this estimator builds the {self._mode} stem"
            )

    def _check_pair(self, reference: RawSensorFrame, source: RawSensorFrame) -> None:
        self._check_frame(reference, "reference")
        self._check_frame(source, "source")
        if reference.packed_shape != source.packed_shape:
            raise ValueError(
                "the two frames must share one packed grid, got "
                f"{reference.packed_shape} vs {source.packed_shape}"
            )
        if reference.dataset_id != source.dataset_id:
            raise ValueError(
                "cross-dataset correspondence is refused: frames come from different "
                "dataset_id values"
            )
        if str(reference.camera_id) != str(source.camera_id):
            raise ValueError(
                "cross-camera correspondence is refused until a camera geometry is accepted"
            )
        if self._mode == "cfa4":
            if reference.cfa_pattern != source.cfa_pattern:
                raise ValueError(
                    f"CFA patterns differ ({reference.cfa_pattern!r} vs "
                    f"{source.cfa_pattern!r}); one sensor has one pattern"
                )
            if reference.channel_phase != source.channel_phase:
                raise ValueError(
                    "the two frames pack their CFA phases in different channel orders "
                    f"({reference.channel_phase!r} vs {source.channel_phase!r})"
                )

    def _effective_iters(self, iters: int | None) -> int:
        if iters is None:
            return self._iters
        if isinstance(iters, bool) or not isinstance(iters, int) or iters < 0:
            raise ValueError("iters must be a non-negative int")
        return iters

    def _spec_for(self, frame: RawSensorFrame) -> RawNormalization | None:
        spec = self._normalization
        if spec is None or isinstance(spec, RawNormalization):
            return spec
        try:
            return spec[frame.calibration_version]
        except KeyError as error:
            raise KeyError(
                f"no RawNormalization is declared for calibration_version "
                f"{frame.calibration_version!r}; unknown calibration is not silently defaulted"
            ) from error

    def _normalization_source(self, reference: RawSensorFrame, source: RawSensorFrame) -> str:
        specs = (self._spec_for(reference), self._spec_for(source))
        if specs[0] is None and specs[1] is None:
            return "caller_pre_normalized"
        labels = ["caller_pre_normalized" if spec is None else spec.source for spec in specs]
        if labels[0] == labels[1]:
            return labels[0]
        return f"reference={labels[0]};source={labels[1]}"

    @contextmanager
    def _eval_mode(self):
        was_training = self._model.training
        self._model.eval()
        try:
            yield
        finally:
            self._model.train(was_training)

    def _extract_flow(
        self, outputs: Mapping[str, Any], packed_shape: tuple[int, int]
    ) -> torch.Tensor:
        if not isinstance(outputs, Mapping) or "final" not in outputs:
            raise TypeError(
                "the motion model must return a mapping with a 'final' flow, as the vendored "
                "RAFT does"
            )
        flow = outputs["final"]
        if not isinstance(flow, torch.Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
            raise TypeError(f"the 'final' flow must be [N, 2, Hp, Wp], got {type(flow).__name__}")
        if tuple(flow.shape[0:1] + flow.shape[2:4]) != (1, *packed_shape):
            raise ValueError(
                f"the motion model returned flow on {tuple(flow.shape)} instead of the packed "
                f"grid {(1, 2, *packed_shape)}"
            )
        return flow[0].to(torch.float64)

    def _proxy_scores(
        self, outputs: Mapping[str, Any], packed_shape: tuple[int, int]
    ) -> Mapping[str, torch.Tensor] | None:
        """Declared proxy scores from the author mixture head's ``info`` tensor.

        ``info`` holds ``[weight(2), log_b(2)]``.  Clamping matches the author's
        training loss (component 0 to ``[0, var_max]``, component 1 to
        ``[var_min, 0]``), so each component is the scale of a Laplace term.
        ``flow_risk`` is the mixture-mean scale (sensor pixels) and ``visibility``
        is the mixture weight on the low-scale component.  Both are proxies.
        """
        info = outputs.get("info") if isinstance(outputs, Mapping) else None
        if info is None:
            return None
        if isinstance(info, (list, tuple)):
            info = info[-1]
        if not isinstance(info, torch.Tensor) or info.ndim != 4 or info.shape[1] != 4:
            raise TypeError("the model's 'info' must be [N, 4, Hp, Wp] (weight 2 + log_b 2)")
        if tuple(info.shape[2:4]) != packed_shape:
            raise ValueError(
                f"the model's 'info' returned {tuple(info.shape[2:4])} instead of {packed_shape}"
            )
        weight = info[:, :2]
        log_b = info[:, 2:]
        scale_large = torch.exp(torch.clamp(log_b[:, 0], min=0.0, max=self._var_max))
        scale_small = torch.exp(torch.clamp(log_b[:, 1], min=self._var_min, max=0.0))
        mixture = torch.softmax(weight, dim=1)
        scale_packed = (mixture[:, 0] * scale_large + mixture[:, 1] * scale_small)[0]
        return {
            "flow_risk": (scale_packed * SENSOR_PER_PACKED[self._mode]).detach(),
            "visibility": mixture[0, 1].detach(),
        }

    def _calibrate(
        self, proxy: Mapping[str, torch.Tensor] | None
    ) -> tuple[str, torch.Tensor | None, torch.Tensor | None]:
        table = self._risk_calibration
        if table is None:
            return ("proxy" if proxy is not None else "unavailable", None, None)
        if proxy is None:
            raise ValueError(
                "a risk calibration table was supplied but the motion model returned no proxy "
                "scores to calibrate"
            )
        calibrated_risk, coverage = table.map_proxy(
            flow_risk=proxy["flow_risk"], visibility=proxy["visibility"]
        )
        for name, value in (("calibrated_risk", calibrated_risk), ("coverage", coverage)):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(
                proxy["flow_risk"].shape
            ):
                raise ValueError(
                    f"{name} must be a tensor on the packed grid {tuple(proxy['flow_risk'].shape)}"
                )
        return f"calibrated:{table.version}", calibrated_risk, coverage

    def _build_taps(
        self,
        source: RawSensorFrame,
        valid: torch.Tensor,
        tap_rows: tuple[torch.Tensor, ...],
        tap_cols: tuple[torch.Tensor, ...],
        tap_weights: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[SourceKey, ...], tuple[float, ...]]:
        grid = tap_rows[0]
        keys: list[SourceKey] = []
        weights: list[float] = []
        rows = [tap.tolist() for tap in tap_rows]
        cols = [tap.tolist() for tap in tap_cols]
        values = [tap.tolist() for tap in tap_weights]
        for channel, row, col in valid.nonzero(as_tuple=False).tolist():
            for tap in range(4):
                weight = values[tap][row][col]
                if weight == 0.0:
                    continue
                i = int(rows[tap][row][col])
                j = int(cols[tap][row][col])
                if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]):
                    raise AssertionError(
                        "an out-of-frame tap passed the validity mask; the sampling contract "
                        "is broken"
                    )
                keys.append(source.sample_key(channel, i, j))
                weights.append(weight)
        return tuple(keys), tuple(weights)


class _SeaRaftArgs:
    """The author argument surface the vendored modules read, at their M sizes."""

    #: Author `RAFT.__init__` forces this many correlation pyramid levels.
    CORR_LEVELS = 4

    def __init__(self, *, iters: int) -> None:
        self.initial_dim = 64
        self.block_dims = [64, 128, 256]
        self.dim = 128
        self.radius = 4
        self.num_blocks = 2
        self.iters = int(iters)
        self.corr_levels = self.CORR_LEVELS
        self.use_var = True
        self.var_min = 0.0
        self.var_max = 10.0

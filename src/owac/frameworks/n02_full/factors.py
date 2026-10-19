"""P1 conditional measurement factors (T3).

Frozen interface: ``docs/prompts/n02_v1_1/p1_interface_freeze.md`` sections 3.4
and 3.7. This module owns exactly three things:

* :class:`ObservationModel` -- the façade that encodes a bundle (delegated to the
  T1 raw observation encoder), predicts one candidate world from an explicitly
  declared history, and scores a prediction against the observed query.
* :class:`CandidateDictionary` / :class:`CandidateSpec` / :class:`CandidateGeometry`
  -- the public controlled-task dictionary. A candidate carries only a geometric
  hypothesis and a declared nuisance hypothesis; the structural guard below makes
  it impossible to smuggle the current query, a truth label or an arbitrary skip
  into a candidate, and every candidate is encoded by the same shared decoder.
* :class:`ConditionalRawDecoder` -- one shared decoder conditioned on the pooled
  history context and the candidate embedding. It emits a low-resolution grid,
  progressively upsamples it and samples it at the query coordinates, returning a
  mean and a positively parameterised variance per query point.

The fit / prequential split is a hard behavioural boundary: ``predict`` with
``call="prequential_check"`` refuses any history that reaches the query frame,
while ``call="fit"`` may use the explicit measurement set (including the query
frame). A first frame without history is marked ``calibration_only`` and never
counts as an independent check. Nuisance ``eta`` enters only through the declared
``HistoryContext.eta``; this module never reads ``EvaluationTruth``.
:class:`AnalyticOracle` is a separate class with the same predict/score interface
for the evaluator's information upper bound and is never a dictionary member.

Query points are always a ``(camera, query-point)`` pairing: ``QuerySpec`` and
``RawQuery`` carry the per-point camera index, ``predict`` returns it in
``RawPrediction.masks["camera_index"]``, and :meth:`ObservationModel.score`
refuses to broadcast a prediction across another camera's points
(:class:`QueryAlignmentError`). Cameras of different resolutions simply contribute
different point counts; use :meth:`QuerySpec.select_camera` /
:meth:`RawQuery.select_camera` to score one camera at a time and sum the scores
into the one objective.

Learnable features keep their graph: ``pooled_features`` and a pyramid level's
``values`` are only re-typed to float32 (never detached) so the raw encoder keeps
its gradient path, while declared data (measurements, candidate/truth geometry) is
detached as before.

``build_p1_task`` is the generator-side assembly helper: it is the *only* place
where truth geometry may shape a candidate dictionary, and the resulting
``TrainingTargets.optional_geometry`` carries the constructive candidate label for
the training harness (loss stays in T4, never in this module).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from owac.frameworks.n02_full.state import (
    EntityState,
    FeatureLevel,
    InferenceInput,
    MeasurementKey,
    TrainingTargets,
    WorldHypothesis,
)

__all__ = [
    "AnalyticOracle",
    "CALLS",
    "CANDIDATE_CATEGORIES",
    "CandidateDictionary",
    "CandidateEncoder",
    "CandidateGeometry",
    "CandidateSpec",
    "CausalLeakError",
    "QueryAlignmentError",
    "ConditionalRawDecoder",
    "FORBIDDEN_CANDIDATE_KEYS",
    "HistoryContext",
    "MIN_VARIANCE",
    "ObservationModel",
    "P1Task",
    "QuerySpec",
    "RawQuery",
    "UnitAffine",
    "build_p1_task",
    "coerce_raw_query",
    "pool_feature_pyramid",
    "world_positions",
]

#: The two declared call modes of :meth:`ObservationModel.predict`.
CALLS = ("fit", "prequential_check")
#: The two per-candidate categories of the controlled task. ``all_wrong`` is a
#: property of the whole dictionary (no candidate is the true explanation).
CANDIDATE_CATEGORIES = ("correct", "wrong")
#: Fields of the declared geometry hypothesis, in canonical vector order.
GEOMETRY_FIELDS = ("world_px", "scale_px", "depth")
#: Fields of the declared nuisance hypothesis, in canonical vector order.
NUISANCE_FIELDS = ("log_exposure", "camera_shift_px", "noise_scale")
#: Candidate field names that could copy the current query or a truth label.
#: Declaring one is a hard error: that is the structural "no skip" guard.
FORBIDDEN_CANDIDATE_KEYS = frozenset(
    {
        "answer",
        "answers",
        "measurement",
        "measurements",
        "query",
        "query_values",
        "raw",
        "raw_query",
        "raw_values",
        "residual",
        "residuals",
        "skip",
        "skip_values",
        "target",
        "targets",
        "truth",
    }
)

#: Positive floor of the predicted variance.
MIN_VARIANCE = 1e-6
#: Default declared offsets of the wrong explanations of a controlled task.
DEFAULT_WRONG_OFFSETS = ((2.0, 1.0), (-3.0, 2.0), (1.0, -4.0))


class CausalLeakError(ValueError):
    """A prequential query was given a history that reaches or covers it."""


class QueryAlignmentError(ValueError):
    """A prediction and a query disagree on their (camera, query-point) pairing.

    Predictions and queries are paired point by point; they are never broadcast
    along the query-point axis, because one camera's prediction must not be
    broadcast onto another camera's samples.
    """


@dataclass(slots=True, frozen=True)
class UnitAffine:
    """Affine map from the model's internal prediction domain to declared units.

    ``units_value = internal_value * scale + offset`` and
    ``units_variance = internal_variance * scale²``. The parameters are the
    declared calibration affine (for example ``scale = white - black``,
    ``offset = black``), recorded together with the calibration version so a
    manifest can state the whole unit chain. RawPredictions are always emitted in
    the declared query units: an internal (for example normalized) prediction
    domain is inverted before anything is scored.
    """

    scale: float
    offset: float = 0.0
    units: str = ""
    calibration_version: str = ""

    def __post_init__(self) -> None:
        scale = float(self.scale)
        offset = float(self.offset)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"unit affine scale must be finite and positive, got {scale!r}")
        if not math.isfinite(offset):
            raise ValueError(f"unit affine offset must be finite, got {offset!r}")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "units", str(self.units))
        object.__setattr__(self, "calibration_version", str(self.calibration_version))

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.offset == 0.0

    def to_units(
        self, mean: torch.Tensor, variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert the internal normalization; the variance scales by ``scale²``."""
        return mean * self.scale + self.offset, variance * (self.scale * self.scale)


def _cfg(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_float_tensor(
    value: Any, name: str, *, detach: bool = True, check_finite: bool = True
) -> torch.Tensor:
    """Float32 view of ``value``.

    ``detach=False`` is what the learnable feature path uses: an existing tensor
    is only re-typed/re-deviced, so the raw encoder keeps its gradient path (the
    P1 §5 Base lr schedule). Declared data (measurements, candidate geometry,
    truth geometry) keeps the default detaching conversion.

    ``check_finite=False`` is the hot-path setting: ``isfinite().all()`` is a
    device-wide reduction that synchronizes CUDA, and the measurement step calls
    this once per candidate per row. Declared data (built once) keeps the check;
    values that are re-read every step (pooled features, the geometry of an
    already validated candidate) skip it.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach() if detach else value
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        except (TypeError, ValueError) as error:  # pragma: no cover - defensive
            raise TypeError(f"{name} must be a float tensor-like value") from error
    if check_finite and not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _normalise_truth_geometry(truth_geometry: Any) -> torch.Tensor | None:
    """``[N, 2]`` truth positions from a mapping, a truth record or a tensor."""
    if truth_geometry is None:
        return None
    candidate: Any = None
    if isinstance(truth_geometry, Mapping):
        candidate = truth_geometry.get("world_px")
        inner = truth_geometry.get("geometry")
        if candidate is None and isinstance(inner, Mapping):
            candidate = inner.get("world_px")
    else:
        candidate = getattr(truth_geometry, "world_px", None)
        if candidate is None:
            geometry = getattr(truth_geometry, "geometry", None)
            if isinstance(geometry, Mapping):
                candidate = geometry.get("world_px")
        if candidate is None:
            candidate = truth_geometry  # a bare tensor/array of positions
    if candidate is None:
        return None
    try:
        tensor = _as_float_tensor(candidate, "truth_geometry.world_px")
    except (TypeError, ValueError):
        return None
    if tensor.dim() != 2 or tensor.shape[-1] != 2:
        return None
    return tensor


def world_positions(world: WorldHypothesis) -> torch.Tensor | None:
    """``[N, 2]`` view geometry of a world, or ``None`` when it declares none."""
    entities = getattr(world, "entities", ())
    if not entities:
        return None
    geometry = getattr(entities[0], "geometry", None)
    if geometry is None:
        return None
    if isinstance(geometry, Mapping):
        positions = geometry.get("world_px", geometry.get("positions_px"))
    else:
        positions = getattr(geometry, "world_px", getattr(geometry, "positions_px", None))
    if positions is None:
        return None
    try:
        tensor = _as_float_tensor(positions, "world geometry world_px", check_finite=False)
    except (TypeError, ValueError):
        return None
    if tensor.dim() != 2 or tensor.shape[-1] != 2:
        raise ValueError("world geometry world_px must have shape [N, 2]")
    return tensor


def _world_geometry_field(world: WorldHypothesis, name: str, default: torch.Tensor) -> torch.Tensor:
    """One declared vector geometry field of a world with a declared default."""
    entities = getattr(world, "entities", ())
    if not entities:
        return default
    geometry = getattr(entities[0], "geometry", None)
    if geometry is None:
        return default
    value = geometry.get(name) if isinstance(geometry, Mapping) else getattr(geometry, name, None)
    if value is None:
        return default
    tensor = _as_float_tensor(value, f"world geometry {name}", check_finite=False)
    if tensor.dim() != 1 or tensor.shape[0] != default.shape[0]:
        raise ValueError(
            f"world geometry {name} must have shape {tuple(default.shape)}, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def _check_declared_keys(declared: Mapping[str, Any], where: str) -> None:
    forbidden = sorted(
        str(key) for key in declared if str(key).strip().lower() in FORBIDDEN_CANDIDATE_KEYS
    )
    if forbidden:
        raise ValueError(
            f"{where} declares fields that could copy the current query or a truth "
            f"label: {forbidden}"
        )


# --------------------------------------------------------------------------------------
# query / history / raw query
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class QuerySpec:
    """The query positions of one decision time; never the measured values.

    ``camera_ids`` gives the camera of every query point (``Q`` entries), so the
    point axis is always a ``(camera, query-point)`` pairing. ``keys`` carries one
    measurement identity per camera, in the same order as the per-point camera
    indices used by :meth:`ObservationModel.predict`'s ``camera_index`` mask.
    ``unit_affine`` declares, per camera, the affine that inverts the model's
    internal prediction domain into :attr:`units` (see :class:`UnitAffine`);
    cameras without an entry are already declared in units.
    """

    keys: tuple[MeasurementKey, ...]
    sensor_coordinates: torch.Tensor
    camera_ids: tuple[str, ...]
    frame_index: int
    units: str
    unit_affine: dict[str, UnitAffine] | None = None
    _camera_index_cache: torch.Tensor | None = None
    _geometry_cache: dict = field(default_factory=dict)

    @classmethod
    def _trusted(
        cls,
        *,
        keys: tuple[MeasurementKey, ...],
        sensor_coordinates: torch.Tensor,
        camera_ids: tuple[str, ...],
        frame_index: int,
        units: str,
        unit_affine: dict[str, UnitAffine] | None = None,
    ) -> QuerySpec:
        """Fast path for specs this package built from already-validated coordinates.

        The samplers construct ``sensor_coordinates`` from integer query grids and
        crop offsets, so they are finite and consistent by construction; validating
        them again would add one device-wide reduction per decision time. Caller
        supplied payloads must go through the normal constructor.
        """
        instance = object.__new__(cls)
        instance.keys = tuple(keys)
        instance.sensor_coordinates = sensor_coordinates
        instance.camera_ids = tuple(str(item) for item in camera_ids)
        instance.frame_index = int(frame_index)
        instance.units = str(units)
        instance.unit_affine = unit_affine
        instance._camera_index_cache = None
        instance._geometry_cache = {}
        return instance

    def __post_init__(self) -> None:
        self._camera_index_cache = None
        self._geometry_cache = {}
        self.keys = tuple(self.keys)
        if not self.keys:
            raise ValueError("a query spec needs at least one measurement key")
        for key in self.keys:
            if not isinstance(key, MeasurementKey):
                raise TypeError(f"query keys must be MeasurementKey, got {type(key).__name__}")
        key_cameras = [key.camera_id for key in self.keys]
        if len(set(key_cameras)) != len(key_cameras):
            raise ValueError(f"one camera may hold one query key, got {key_cameras}")
        coordinates = _as_float_tensor(self.sensor_coordinates, "sensor_coordinates")
        if coordinates.dim() != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"sensor_coordinates must be (Q, 2), got {tuple(coordinates.shape)}")
        self.sensor_coordinates = coordinates
        self.camera_ids = tuple(str(item) for item in self.camera_ids)
        if len(self.camera_ids) != coordinates.shape[0]:
            raise ValueError(
                f"camera_ids has {len(self.camera_ids)} entries but there are "
                f"{coordinates.shape[0]} query points"
            )
        declared = set(self.camera_ids)
        missing = sorted(set(key_cameras) - declared)
        if missing:
            raise ValueError(f"query keys refer to cameras absent from camera_ids: {missing}")
        self.frame_index = int(self.frame_index)
        self.units = str(self.units)
        if self.unit_affine is not None:
            affines: dict[str, UnitAffine] = {}
            for camera, affine in dict(self.unit_affine).items():
                camera = str(camera)
                if not isinstance(affine, UnitAffine):
                    raise TypeError(
                        f"unit_affine[{camera!r}] must be a UnitAffine, got {type(affine).__name__}"
                    )
                if affine.units and affine.units != self.units:
                    raise ValueError(
                        f"unit_affine[{camera!r}] declares units {affine.units!r} but the "
                        f"query declares {self.units!r}"
                    )
                affines[camera] = affine
            unknown = sorted(set(affines) - declared)
            if unknown:
                raise ValueError(f"unit_affine refers to cameras outside the query: {unknown}")
            self.unit_affine = affines

    def affine_tensors(
        self, *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Q,)`` per-point scale and offset from the internal domain to units.

        Built with tensor scatter from the cached camera index: the hot path has no
        per-point Python loop.
        """
        key = ("affine", str(device))
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        count = self.num_points
        scale = torch.ones(count, dtype=torch.float32, device=device)
        offset = torch.zeros(count, dtype=torch.float32, device=device)
        if self.unit_affine:
            index = self.camera_index().to(device)
            for camera, affine in self.unit_affine.items():
                if affine.is_identity:
                    continue
                position = torch.nonzero(index == self.key_position(camera), as_tuple=False)
                if position.numel() == 0:
                    continue
                scale[position] = affine.scale
                offset[position] = affine.offset
        self._geometry_cache[key] = (scale, offset)
        return scale, offset

    def key_position(self, camera_id: str) -> int:
        """Declared key position of one camera (``O(V)``, no per-point work)."""
        camera = str(camera_id)
        for position, key in enumerate(self.keys):
            if key.camera_id == camera:
                return position
        raise KeyError(
            f"camera {camera!r} is not part of this query; "
            f"have {[key.camera_id for key in self.keys]}"
        )

    def coordinate_divisors(
        self, sensor_shape: Mapping[str, Any], *, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Q,)`` per-point sensor height/width, cached per ``(shapes, device)``.

        The divisors are assembled by one gather from a per-camera table, so the
        coordinate normalization for a full-resolution query is a handful of tensor
        operations instead of one Python step per sample point.
        """
        key = (
            tuple(sorted((str(c), int(v[0]), int(v[1])) for c, v in sensor_shape.items())),
            str(device),
        )
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        heights: list[float] = []
        widths: list[float] = []
        for measurement_key in self.keys:
            camera = measurement_key.camera_id
            if camera not in sensor_shape:
                raise KeyError(
                    f"config.sensor_shape does not declare camera {camera!r}; sensor pixel "
                    "coordinates cannot be normalized without it"
                )
            shape = sensor_shape[camera]
            if len(shape) != 2 or int(shape[0]) < 1 or int(shape[1]) < 1:
                raise ValueError(f"sensor shape for {camera!r} must be (H, W), got {shape}")
            heights.append(float(shape[0]))
            widths.append(float(shape[1]))
        index = self.camera_index()
        height_tensor = torch.tensor(heights, dtype=torch.float32)[index]
        width_tensor = torch.tensor(widths, dtype=torch.float32)[index]
        if device is not None:
            height_tensor = height_tensor.to(device)
            width_tensor = width_tensor.to(device)
        self._geometry_cache[key] = (height_tensor, width_tensor)
        return height_tensor, width_tensor

    @property
    def num_points(self) -> int:
        return int(self.sensor_coordinates.shape[0])

    def camera_index(self, device: torch.device | None = None) -> torch.Tensor:
        """``(Q,)`` long index into :attr:`keys` for every query point (cached).

        The array is built once per spec (and once per device), so repeated
        ``predict``/``score`` calls of one decision time rebuild neither the array
        nor its device copy; prediction and query then share one tensor.
        """
        cached = self._camera_index_cache
        if cached is None:
            order = {key.camera_id: position for position, key in enumerate(self.keys)}
            cached = torch.tensor([order[camera] for camera in self.camera_ids], dtype=torch.long)
            self._camera_index_cache = cached
        if device is None or cached.device == device:
            return cached
        key = ("camera_index", str(device))
        moved = self._geometry_cache.get(key)
        if moved is None:
            moved = cached.to(device)
            self._geometry_cache[key] = moved
        return moved

    def affine_is_identity(self) -> bool:
        """Whether no camera declares a non-identity internal->units affine (cached)."""
        cached = self._geometry_cache.get("affine_identity")
        if cached is None:
            cached = all(affine.is_identity for affine in (self.unit_affine or {}).values())
            self._geometry_cache["affine_identity"] = cached
        return bool(cached)

    def camera_columns(self, camera_order: Sequence[str]) -> torch.Tensor:
        """``(Q,)`` long column index of every point in ``camera_order`` (tensor gather)."""
        column_of_key = {str(camera): position for position, camera in enumerate(camera_order)}
        table = torch.tensor([column_of_key[key.camera_id] for key in self.keys], dtype=torch.long)
        return table[self.camera_index()]

    def camera_point_counts(self) -> dict[str, int]:
        """Per-camera query-point counts; cameras may differ (different resolutions)."""
        counts = torch.bincount(self.camera_index(), minlength=len(self.keys))
        return {key.camera_id: int(counts[position]) for position, key in enumerate(self.keys)}

    def camera_positions(self, camera_id: str) -> torch.Tensor:
        """``(q_camera,)`` long positions of one camera's points in the point axis."""
        camera = str(camera_id)
        order = {key.camera_id: position for position, key in enumerate(self.keys)}
        if camera not in order:
            raise KeyError(f"camera {camera!r} is not part of this query; have {sorted(order)}")
        matches = torch.nonzero(self.camera_index() == order[camera], as_tuple=False)
        return matches.reshape(-1)

    def select_camera(self, camera_id: str) -> QuerySpec:
        """A spec holding only one camera's ``(camera, point)`` pairs."""
        camera = str(camera_id)
        positions = self.camera_positions(camera)
        if positions.numel() == 0:
            raise KeyError(f"camera {camera!r} holds no query points")
        keys = tuple(key for key in self.keys if key.camera_id == camera)
        return QuerySpec._trusted(
            keys=keys,
            sensor_coordinates=self.sensor_coordinates.index_select(0, positions),
            camera_ids=tuple(camera for _ in range(int(positions.numel()))),
            frame_index=self.frame_index,
            units=self.units,
            unit_affine=(
                {camera: self.unit_affine[camera]}
                if self.unit_affine and camera in self.unit_affine
                else None
            ),
        )


@dataclass(slots=True)
class HistoryContext:
    """The declared measurement context of one prediction.

    ``pooled_features`` is ``(B, feature_dim)``; ``last_frame_index`` is ``None``
    when there is no history at all (first frame -> calibration-only). ``eta`` is
    the declared nuisance estimate (``InterferenceEstimate`` from ``sensor.py``)
    and ``eta_fit_keys`` names the measurements that eta was fitted on, so a query
    that participated in the eta fit can be marked ``empirical_check``.
    """

    pooled_features: torch.Tensor
    levels: dict[str, FeatureLevel] | None = None
    episode_ids: tuple[str, ...] = ()
    anchor_ids: tuple[str, ...] = ()
    last_frame_index: int | None = None
    eta: Any = None
    eta_fit_keys: tuple[MeasurementKey, ...] = ()

    def __post_init__(self) -> None:
        self.episode_ids = tuple(str(item) for item in self.episode_ids)
        self.anchor_ids = tuple(str(item) for item in self.anchor_ids)
        if self.levels is not None and not isinstance(self.levels, dict):
            raise TypeError(f"levels must be a dict, got {type(self.levels).__name__}")
        self.eta_fit_keys = tuple(self.eta_fit_keys)
        for key in self.eta_fit_keys:
            if not isinstance(key, MeasurementKey):
                raise TypeError(f"eta_fit_keys must be MeasurementKey, got {type(key).__name__}")
        if self.pooled_features is not None and not isinstance(self.pooled_features, torch.Tensor):
            self.pooled_features = _as_float_tensor(self.pooled_features, "pooled_features")
        if self.last_frame_index is not None:
            self.last_frame_index = int(self.last_frame_index)


@dataclass(slots=True)
class RawQuery:
    """Observed query values aligned with :class:`QuerySpec` sampling points.

    ``values`` is ``(B, Q)`` or ``(Q,)``; ``masks`` holds boolean ``valid`` /
    ``bad`` / ``missing`` / ``saturated`` tensors with the same shape. ``units``
    must equal :attr:`ObservationModel` prediction units verbatim.
    ``camera_index`` is the ``(Q,)``/``(B, Q)`` long index into ``keys`` of every
    query point: it is what makes a multi-camera concatenation explicit, and
    :meth:`select_camera` is the supported way to score one camera at a time.
    ``unit_affine`` mirrors :class:`QuerySpec`: when declared, a prediction scored
    against this query must state the unit transform it was built with.
    """

    values: torch.Tensor
    masks: dict[str, Any] = field(default_factory=dict)
    units: str = ""
    keys: tuple[MeasurementKey, ...] = ()
    camera_index: torch.Tensor | None = None
    unit_affine: dict[str, UnitAffine] | None = None

    def __post_init__(self) -> None:
        self.values = _as_float_tensor(self.values, "raw query values")
        if self.values.dim() not in (1, 2):
            raise ValueError(f"raw query values must be (Q,) or (B, Q), got {self.values.shape}")
        self.masks = {
            str(name): torch.as_tensor(value).bool() for name, value in dict(self.masks).items()
        }
        self.units = str(self.units)
        self.keys = tuple(self.keys)
        if self.unit_affine is not None:
            affines: dict[str, UnitAffine] = {}
            for camera, affine in dict(self.unit_affine).items():
                if not isinstance(affine, UnitAffine):
                    raise TypeError(
                        f"unit_affine[{camera!r}] must be a UnitAffine, got {type(affine).__name__}"
                    )
                if affine.units and affine.units != self.units:
                    raise ValueError(
                        f"unit_affine[{camera!r}] declares units {affine.units!r} but the "
                        f"query declares {self.units!r}"
                    )
                affines[str(camera)] = affine
            self.unit_affine = affines
        if self.camera_index is not None:
            index = torch.as_tensor(self.camera_index, dtype=torch.long)
            if index.shape[-1] != self.num_points or index.dim() > 2:
                raise ValueError(
                    f"camera_index must be (Q,) or (B, Q) with Q={self.num_points}, got "
                    f"{tuple(index.shape)}"
                )
            if self.keys and index.numel():
                # One synchronization at most: range and negativity in one reduction
                # (this constructor runs per decision time, so extra syncs are the
                # difference between a seconds and a minutes step on a shared GPU).
                invalid = bool(((index < 0) | (index >= len(self.keys))).any().item())
                if invalid:
                    raise ValueError(
                        f"camera_index must reference one of {len(self.keys)} keys and be "
                        f"non-negative; got {index.detach().cpu().tolist()}"
                    )
            self.camera_index = index

    @classmethod
    def _trusted(
        cls,
        *,
        values: torch.Tensor,
        masks: dict[str, Any],
        units: str,
        keys: tuple[MeasurementKey, ...] = (),
        camera_index: torch.Tensor | None = None,
        unit_affine: dict[str, UnitAffine] | None = None,
    ) -> RawQuery:
        """Fast path for objects this module built and already validated once.

        The samplers (:func:`owac.frameworks.n02_full.model.build_raw_query`,
        :meth:`select_camera`, the device-aligning move in ``score``) produce
        correctly shaped, non-negative indices by construction; re-validating them
        on every score call would add a device synchronization per call. Never use
        this for caller-supplied payloads: use the normal constructor.
        """
        instance = object.__new__(cls)
        instance.values = values
        instance.masks = masks
        instance.units = units
        instance.keys = tuple(keys)
        instance.camera_index = camera_index
        instance.unit_affine = unit_affine
        return instance

    @property
    def num_points(self) -> int:
        return int(self.values.shape[-1])

    def camera_point_counts(self) -> dict[str, int]:
        """Per-camera query-point counts from the declared camera index."""
        if self.camera_index is None:
            if len(self.keys) == 1:
                return {self.keys[0].camera_id: self.num_points}
            return {}
        index = self.camera_index.reshape(-1)
        return {
            key.camera_id: int((index == position).sum()) for position, key in enumerate(self.keys)
        }

    def select_camera(self, camera_id: str) -> RawQuery:
        """A query holding only one camera's values/masks/points."""
        camera = str(camera_id)
        if self.camera_index is None:
            if len(self.keys) == 1 and self.keys[0].camera_id == camera:
                return self
            raise QueryAlignmentError(
                f"this raw query carries no camera index and its keys are "
                f"{[key.camera_id for key in self.keys]}; it cannot be sliced per camera"
            )
        order = {key.camera_id: position for position, key in enumerate(self.keys)}
        if camera not in order:
            raise KeyError(f"camera {camera!r} is not part of this query; have {sorted(order)}")
        index = self.camera_index.reshape(-1)
        positions = torch.nonzero(index == order[camera], as_tuple=False).reshape(-1)
        if positions.numel() == 0:
            raise KeyError(f"camera {camera!r} holds no query points")

        def take(value: torch.Tensor) -> torch.Tensor:
            if value.dim() >= 1 and int(value.shape[-1]) == self.num_points:
                return value.index_select(value.dim() - 1, positions)
            return value

        return RawQuery._trusted(
            values=take(self.values),
            masks={name: take(value) for name, value in self.masks.items()},
            units=self.units,
            keys=(tuple(key for key in self.keys if key.camera_id == camera)[0],),
            camera_index=torch.zeros(
                int(positions.numel()), dtype=torch.long, device=self.values.device
            ),
            unit_affine=(
                {camera: self.unit_affine[camera]}
                if self.unit_affine and camera in self.unit_affine
                else None
            ),
        )


def coerce_raw_query(raw_query: Any) -> RawQuery:
    """Normalize a raw query payload into :class:`RawQuery`."""
    if isinstance(raw_query, RawQuery):
        return raw_query
    if isinstance(raw_query, Mapping):
        values = raw_query.get("values")
        if values is None:
            raise TypeError("raw query mapping must carry 'values'")
        return RawQuery(
            values=values,
            masks=dict(raw_query.get("masks", {})),
            units=str(raw_query.get("units", "")),
            keys=tuple(raw_query.get("keys", ())),
            camera_index=raw_query.get("camera_index"),
            unit_affine=raw_query.get("unit_affine"),
        )
    values = getattr(raw_query, "values", None)
    if values is None:
        raise TypeError(
            "raw_query must be a RawQuery, a mapping with 'values', or an object with "
            f"'values'; got {type(raw_query).__name__}"
        )
    return RawQuery(
        values=values,
        masks=dict(getattr(raw_query, "masks", None) or {}),
        units=str(getattr(raw_query, "units", "")),
        keys=tuple(getattr(raw_query, "keys", None) or ()),
        camera_index=getattr(raw_query, "camera_index", None),
        unit_affine=getattr(raw_query, "unit_affine", None),
    )


# --------------------------------------------------------------------------------------
# candidate dictionary
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class CandidateGeometry:
    """A geometric hypothesis only: view positions plus optional scale and depth."""

    world_px: Any
    scale_px: Any = None
    depth: Any = None

    def __post_init__(self) -> None:
        positions = _as_float_tensor(self.world_px, "candidate world_px")
        if positions.dim() != 2 or positions.shape[1] != 2:
            raise ValueError(f"candidate world_px must be (N, 2), got {tuple(positions.shape)}")
        if positions.shape[0] < 1:
            raise ValueError("a candidate geometry needs at least one object")
        self.world_px = positions
        count = int(positions.shape[0])
        self.scale_px = (
            torch.ones(count)
            if self.scale_px is None
            else _as_float_tensor(self.scale_px, "candidate scale_px")
        )
        self.depth = (
            torch.zeros(count)
            if self.depth is None
            else _as_float_tensor(self.depth, "candidate depth")
        )
        for name in ("scale_px", "depth"):
            value = getattr(self, name)
            if value.dim() != 1 or value.shape[0] != count:
                raise ValueError(f"candidate {name} must have shape ({count},)")

    @property
    def world_size(self) -> int:
        return int(self.world_px.shape[0])

    def vector(self) -> torch.Tensor:
        """Canonical flat geometry vector; identical width for every candidate."""
        return torch.cat(
            [
                self.world_px.reshape(-1),
                self.scale_px.reshape(-1),
                self.depth.reshape(-1),
            ]
        )


@dataclass(slots=True)
class CandidateSpec:
    """One declared candidate: identity, category, geometry and nuisance."""

    candidate_id: str
    category: str
    geometry: CandidateGeometry
    nuisance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.candidate_id = str(self.candidate_id)
        if not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if self.category not in CANDIDATE_CATEGORIES:
            raise ValueError(
                f"candidate category must be one of {CANDIDATE_CATEGORIES}, got {self.category!r}"
            )
        if not isinstance(self.geometry, CandidateGeometry):
            raise TypeError(
                f"candidate geometry must be a CandidateGeometry, got "
                f"{type(self.geometry).__name__}"
            )
        if not isinstance(self.nuisance, dict):
            raise TypeError(
                f"candidate nuisance must be a dict, got {type(self.nuisance).__name__}"
            )
        _check_declared_keys(self.nuisance, f"candidate {self.candidate_id!r} nuisance")
        unknown = sorted(set(self.nuisance) - set(NUISANCE_FIELDS))
        if unknown:
            raise ValueError(
                f"candidate {self.candidate_id!r} declares unknown nuisance fields {unknown}; "
                f"declared fields are {NUISANCE_FIELDS}"
            )
        self.nuisance = dict(self.nuisance)

    def as_world(self) -> WorldHypothesis:
        """The world hypothesis handed to :meth:`ObservationModel.predict`."""
        geometry = {
            "world_px": self.geometry.world_px.clone(),
            "scale_px": self.geometry.scale_px.clone(),
            "depth": self.geometry.depth.clone(),
        }
        entity = EntityState(
            entity_ids=(self.candidate_id,),
            geometry=geometry,
            geometry_frame="view_2d",
        )
        return WorldHypothesis(
            world_id=self.candidate_id,
            entities=(entity,),
            nuisance={"declared": dict(self.nuisance)},
            score=0.0,
            score_kind="normalized_density",
            ancestor_id=None,
            version=0,
        )


@dataclass(slots=True)
class CandidateDictionary:
    """The public dictionary of the controlled task: correct / wrong / all-wrong.

    ``label_index`` is constructive: the correct candidate's index is fixed when
    the dictionary is built (from the declared category, optionally pinned by
    ``correct_index``), never inferred from the truth by nearest neighbour. The
    cross-check against a supplied truth geometry is reported in
    ``last_cross_check`` and may return ``None`` when the declared geometry and
    the truth disagree.
    """

    specs: tuple[CandidateSpec, ...] = ()
    all_wrong: bool = False
    oracle: Any = None
    correct_index: int | None = None
    last_cross_check: dict | None = None

    def __post_init__(self) -> None:
        self.specs = tuple(self.specs)
        if not self.specs:
            raise ValueError("a candidate dictionary needs at least one candidate")
        for spec in self.specs:
            if not isinstance(spec, CandidateSpec):
                raise TypeError(
                    f"dictionary specs must be CandidateSpec, got {type(spec).__name__}"
                )
        identifiers = [spec.candidate_id for spec in self.specs]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"candidate ids must be unique, got {identifiers}")
        world_sizes = {spec.geometry.world_size for spec in self.specs}
        if len(world_sizes) != 1:
            raise ValueError(
                f"every candidate must declare the same world size, got {sorted(world_sizes)}"
            )
        correct = [index for index, spec in enumerate(self.specs) if spec.category == "correct"]
        wrong = [index for index, spec in enumerate(self.specs) if spec.category == "wrong"]
        if self.all_wrong:
            if correct:
                raise ValueError("an all-wrong dictionary must not contain a correct candidate")
            if not wrong:
                raise ValueError("an all-wrong dictionary needs at least one wrong candidate")
        else:
            if not correct:
                raise ValueError(
                    "a dictionary needs at least one declared correct candidate "
                    "(use all_wrong=True for the all-wrong class)"
                )
            if not wrong:
                raise ValueError("a dictionary needs at least one wrong candidate")
            if len(correct) > 1:
                raise ValueError(
                    f"the controlled task declares exactly one correct candidate, got {correct}"
                )
        if self.correct_index is None:
            self.correct_index = correct[0] if (correct and not self.all_wrong) else None
        else:
            self.correct_index = int(self.correct_index)
            if self.all_wrong:
                raise ValueError("an all-wrong dictionary cannot pin a correct index")
            if not 0 <= self.correct_index < len(self.specs):
                raise ValueError(f"correct_index {self.correct_index} is outside the dictionary")
            if self.specs[self.correct_index].category != "correct":
                raise ValueError(
                    f"candidate {self.specs[self.correct_index].candidate_id!r} is declared "
                    f"{self.specs[self.correct_index].category!r}, not 'correct'"
                )

    # -- declared views ------------------------------------------------------------------

    def worlds(self) -> tuple[WorldHypothesis, ...]:
        """Every candidate of the dictionary, in declared order (oracle excluded)."""
        return tuple(spec.as_world() for spec in self.specs)

    def oracle_world(self) -> WorldHypothesis:
        """The analytic oracle world; evaluator-only, never part of :meth:`worlds`."""
        if self.oracle is None:
            raise LookupError(
                "no analytic oracle is attached to this dictionary; the oracle is an "
                "evaluator-only information upper bound and never a dictionary member"
            )
        return self.oracle.world()

    def label_index(self, truth_geometry: Any = None) -> int | None:
        """Constructive index of the correct candidate, or ``None``.

        Without a truth geometry the constructive label is returned as constructed.
        With one, the declared correct candidate's geometry must match it exactly;
        otherwise the label is unavailable (``None``), the nearest-neighbour result
        is recorded in :attr:`last_cross_check` and reported, never used as label.
        """
        if self.correct_index is None:
            return None
        if truth_geometry is None:
            return self.correct_index
        truth = _normalise_truth_geometry(truth_geometry)
        report = self.cross_check(truth)
        self.last_cross_check = report
        if truth is None or not report["exact_match"]:
            return None
        return self.correct_index

    def cross_check(self, truth_geometry: Any = None) -> dict:
        """Honest cross-check report; nearest neighbour is a check, not a label."""
        truth = _normalise_truth_geometry(truth_geometry)
        report: dict[str, Any] = {
            "constructive_index": self.correct_index,
            "truth_provided": truth is not None,
            "exact_match": False,
            "nearest_index": None,
            "nearest_l2": None,
        }
        if truth is None:
            return report
        if self.correct_index is not None:
            declared = self.specs[self.correct_index].geometry.world_px
            report["exact_match"] = bool(
                declared.shape == truth.shape and torch.equal(declared, truth)
            )
        best_index: int | None = None
        best_distance: float | None = None
        for index, spec in enumerate(self.specs):
            declared = spec.geometry.world_px
            if declared.shape != truth.shape:
                continue
            distance = float(torch.linalg.vector_norm(declared - truth))
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        report["nearest_index"] = best_index
        report["nearest_l2"] = best_distance
        return report

    def categories(self) -> dict[str, int]:
        counts = {name: 0 for name in CANDIDATE_CATEGORIES}
        for spec in self.specs:
            counts[spec.category] += 1
        return counts

    def world_size(self) -> int:
        return self.specs[0].geometry.world_size

    def nuisance_dim(self) -> int:
        return 4  # log_exposure(1) + camera_shift_px(2) + noise_scale(1)

    def capacity(self) -> int:
        """Declared flat candidate width; equal for every candidate by construction."""
        return 4 * self.world_size() + self.nuisance_dim()


# --------------------------------------------------------------------------------------
# conditional decoder
# --------------------------------------------------------------------------------------


class CandidateEncoder(nn.Module):
    """One shared encoder for every candidate; never per-candidate parameters."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.embedding_dim),
        )

    def forward(self, candidate_vector: torch.Tensor) -> torch.Tensor:
        return self.net(candidate_vector)


class ConditionalRawDecoder(nn.Module):
    """Shared conditional RAW decoder: context + candidate -> raw mean and variance.

    The decoder has no query-value input at all: its only query-dependent
    argument is the sampling coordinate. A low-resolution grid is generated from
    the pooled history context and the candidate embedding, progressively
    upsampled, and then sampled at the query points.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 512,
        candidate_dim: int = 32,
        channels: int = 32,
        grid: int = 8,
        hidden: int = 64,
        min_variance: float = MIN_VARIANCE,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.candidate_dim = int(candidate_dim)
        self.channels = int(channels)
        self.grid = int(grid)
        self.min_variance = float(min_variance)
        if min(self.feature_dim, self.candidate_dim, self.channels, self.grid) < 1:
            raise ValueError("decoder dimensions must be positive")
        if self.min_variance <= 0.0:
            raise ValueError("min_variance must be positive")
        self.context_proj = nn.Sequential(nn.Linear(self.feature_dim, hidden), nn.GELU())
        self.candidate_proj = nn.Sequential(
            nn.Linear(self.candidate_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        # Fixed normalized coordinate basis of the low-resolution grid: it makes the
        # generated field depend on the query position without ever seeing a query
        # *value*.
        self.grid_conv = nn.Conv2d(2 * hidden + 2, self.channels, kernel_size=3, padding=1)
        rows = torch.linspace(-1.0, 1.0, self.grid)
        grid_y, grid_x = torch.meshgrid(rows, rows, indexing="ij")
        self.register_buffer("grid_positions", torch.stack([grid_x, grid_y]).unsqueeze(0))
        self.refine = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.GELU(),
        )
        self.mean_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.variance_head = nn.Conv2d(self.channels, 1, kernel_size=1)

    def forward(
        self,
        context: torch.Tensor,
        candidate: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``context`` (B, F), ``candidate`` (B, E), ``coordinates`` (B, N, 2) in [-1, 1]."""
        batch = int(context.shape[0])
        if context.shape[-1] != self.feature_dim:
            raise ValueError(
                f"decoder expects feature dim {self.feature_dim}, got {context.shape[-1]}"
            )
        if candidate.shape[0] != batch or candidate.shape[-1] != self.candidate_dim:
            raise ValueError(
                f"candidate embedding must be ({batch}, {self.candidate_dim}), got "
                f"{tuple(candidate.shape)}"
            )
        if coordinates.dim() != 3 or coordinates.shape[0] != batch or coordinates.shape[-1] != 2:
            raise ValueError(f"coordinates must be ({batch}, N, 2), got {tuple(coordinates.shape)}")
        fused = torch.cat([self.context_proj(context), self.candidate_proj(candidate)], dim=-1)
        conditional = fused.view(batch, -1, 1, 1).expand(-1, -1, self.grid, self.grid)
        positions = self.grid_positions.expand(batch, -1, -1, -1)
        grid = F.gelu(self.grid_conv(torch.cat([conditional, positions], dim=1)))
        field = self.refine(grid)
        mean = self.mean_head(field)
        variance = F.softplus(self.variance_head(field)) + self.min_variance
        sample_grid = coordinates.reshape(batch, -1, 1, 2)
        sampled_mean = F.grid_sample(
            mean, sample_grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(batch, -1)
        sampled_variance = F.grid_sample(
            variance, sample_grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(batch, -1)
        return sampled_mean, sampled_variance


# --------------------------------------------------------------------------------------
# observation model façade
# --------------------------------------------------------------------------------------


def _broadcast_camera_index(index: torch.Tensor, shape: torch.Size, where: str) -> torch.Tensor:
    """A per-point camera index is either (Q,) or (B, Q); anything else is a defect."""
    if index.shape == shape:
        return index
    if index.dim() == len(shape) - 1 and index.shape[-1] == shape[-1]:
        return index.unsqueeze(0).expand(shape)
    raise QueryAlignmentError(
        f"{where} camera_index {tuple(index.shape)} does not match the {where} shape {tuple(shape)}"
    )


def _indices_match(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Equality of two camera indices regardless of the device they were built on.

    A prediction's index follows the model's device while a query's index follows
    the measurement device, so a mixed-device step (for example CPU measurements
    with a CUDA model) must not fail on ``torch.equal``'s same-device requirement.
    The comparison itself is unchanged: equal values pass, different values fail.
    """
    if left is right or (left.shape == right.shape and left.data_ptr() == right.data_ptr()):
        return True  # the same tensor: no comparison, hence no CUDA synchronization
    if left.device != right.device:
        right = right.to(left.device)
    return bool(torch.equal(left, right))


class ObservationModel:
    """Façade over the raw observation encoder, the candidate dictionary and the decoder."""

    def __init__(
        self, encoder: Any, sensor_model: Any, config: Any = None, *, version: str
    ) -> None:
        self.encoder = encoder
        self.sensor_model = sensor_model
        self.config = config
        self.version = str(version)
        self.world_size = int(_cfg(config, "candidate_world_size", 4))
        if self.world_size < 1:
            raise ValueError("candidate_world_size must be positive")
        self.feature_dim = int(_cfg(config, "feature_dim", 512))
        self.candidate_embedding_dim = int(_cfg(config, "candidate_embedding_dim", 32))
        self.decoder_grid = int(_cfg(config, "decoder_grid", 8))
        self.decoder_channels = int(_cfg(config, "decoder_channels", 32))
        self.decoder_hidden = int(_cfg(config, "decoder_hidden", 64))
        shapes = _cfg(config, "sensor_shape", None) or _cfg(config, "camera_shapes", None) or {}
        if not isinstance(shapes, Mapping):
            raise TypeError("sensor_shape config must map camera id -> (H, W)")
        self.sensor_shapes = {str(camera): tuple(shape) for camera, shape in shapes.items()}
        self.structure_version = str(_cfg(config, "structure_version", "p1-fixed-entities"))
        self.initial_prior_version = str(_cfg(config, "initial_prior_version", ""))
        self.candidate_encoder = CandidateEncoder(
            4 * self.world_size + 4, self.candidate_embedding_dim, self.decoder_hidden
        )
        self.decoder = ConditionalRawDecoder(
            feature_dim=self.feature_dim,
            candidate_dim=self.candidate_embedding_dim,
            channels=self.decoder_channels,
            grid=self.decoder_grid,
            hidden=self.decoder_hidden,
        )
        self.candidates: CandidateDictionary | None = None
        self.objective_register: Any = None
        # The measurement path predicts once per candidate per row: the conditioning
        # vector of a world is built once and reused (keyed by object identity, with a
        # strong reference so an id can never be recycled under a stale entry).
        self._world_vectors: dict[int, tuple[Any, torch.Tensor]] = {}

    # -- declared capacities -------------------------------------------------------------

    @property
    def candidate_dim(self) -> int:
        return 4 * self.world_size + 4

    def bind_dictionary(self, dictionary: CandidateDictionary) -> CandidateDictionary:
        """Attach the public dictionary; capacities must match exactly."""
        if not isinstance(dictionary, CandidateDictionary):
            raise TypeError(
                f"dictionary must be a CandidateDictionary, got {type(dictionary).__name__}"
            )
        if dictionary.world_size() != self.world_size:
            raise ValueError(
                f"dictionary world size {dictionary.world_size()} does not match the model "
                f"capacity {self.world_size}"
            )
        if dictionary.capacity() != self.candidate_dim:
            raise ValueError(
                f"dictionary capacity {dictionary.capacity()} does not match the model "
                f"candidate dim {self.candidate_dim}"
            )
        self.candidates = dictionary
        self._world_vectors.clear()
        return dictionary

    # -- encode --------------------------------------------------------------------------

    def encode(self, bundle: Any) -> Any:
        """Delegate RAW encoding to the T1 observation encoder."""
        if self.encoder is None:
            raise RuntimeError("ObservationModel was built without a raw observation encoder")
        return self.encoder.encode(bundle)

    # -- public query construction -------------------------------------------------------

    def build_row_query(self, bundle: Any, row: int = 0, **kwargs: Any) -> Any:
        """Public :func:`owac.frameworks.n02_full.model.build_row_query` for this model.

        Returns the :class:`~owac.frameworks.n02_full.model.RowQuery` (spec + sampling
        plan) the training path uses for ``bundle``/``row``; see that function for the
        evaluator-relevant ``QuerySpec`` fields (``keys``, ``sensor_coordinates``,
        ``camera_ids``, ``units``, ``unit_affine``).
        """
        from owac.frameworks.n02_full.model import build_row_query

        return build_row_query(self, bundle, row, **kwargs)

    def build_raw_query(self, row_query: Any, **kwargs: Any) -> Any:
        """Public :func:`owac.frameworks.n02_full.model.build_raw_query`.

        ``model.build_raw_query(model.build_row_query(bundle, row))`` returns the same
        :class:`RawQuery` (values, masks, units, keys, ``camera_index``) the training
        path scores.
        """
        from owac.frameworks.n02_full.model import build_raw_query

        return build_raw_query(row_query, **kwargs)

    # -- predict -------------------------------------------------------------------------

    def predict(
        self,
        world: WorldHypothesis,
        history: HistoryContext,
        query_spec: QuerySpec,
        *,
        call: str = "prequential_check",
    ) -> Any:
        """Predict the query points of ``world`` from ``history`` under ``call``."""
        if call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}, got {call!r}")
        if not isinstance(world, WorldHypothesis):
            raise TypeError(f"world must be a WorldHypothesis, got {type(world).__name__}")
        if not isinstance(history, HistoryContext):
            raise TypeError(f"history must be a HistoryContext, got {type(history).__name__}")
        if not isinstance(query_spec, QuerySpec):
            raise TypeError(f"query_spec must be a QuerySpec, got {type(query_spec).__name__}")
        if call == "prequential_check":
            self._assert_history_is_causal(history, query_spec)
        self._assert_eta_declared(history)
        context, no_history = self._context_tensor(history)
        device = next(self.decoder.parameters()).device
        if context.device != device:
            context = context.to(device)
        batch = int(context.shape[0])
        vector = self._world_vector(world).to(device=device)
        embedding = self.candidate_encoder(vector.unsqueeze(0)).expand(batch, -1)
        shift = self._eta_translation(history.eta, query_spec, batch)
        if shift is not None and shift.device != device:
            shift = shift.to(device)
        coordinates = self._normalized_coordinates(query_spec, shift, batch, device=device).flip(-1)
        mean, variance = self.decoder(context, embedding, coordinates)
        mean, variance = self._apply_eta_scale(history.eta, mean, variance)
        # Predictions leave this method in the query's declared units: the internal
        # (normalized) domain is inverted here, variance scaling by scale².
        mean, variance, unit_scale, unit_offset = self._apply_unit_affine(
            mean, variance, query_spec, device=device
        )
        calibration_only = call == "prequential_check" and no_history
        empirical = self._eta_seen_query(history, query_spec)
        oracle_eta = (
            history.eta is not None and getattr(history.eta, "source", "history") == "oracle"
        )
        masks: dict[str, torch.Tensor] = {
            "valid": torch.ones_like(mean, dtype=torch.bool),
            "bad": torch.zeros_like(mean, dtype=torch.bool),
            "missing": torch.zeros_like(mean, dtype=torch.bool),
            "saturated": torch.zeros_like(mean, dtype=torch.bool),
            "calibration_only": torch.full_like(mean, bool(calibration_only), dtype=torch.bool),
            "empirical_check": torch.full_like(mean, bool(empirical), dtype=torch.bool),
            "oracle_eta": torch.full_like(mean, bool(oracle_eta), dtype=torch.bool),
            # (camera, query-point) pairing: one long index into query_spec.keys per point
            "camera_index": query_spec.camera_index().to(device).unsqueeze(0).expand(batch, -1),
            "unit_scale": unit_scale,
            "unit_offset": unit_offset,
            "units_declared": torch.ones_like(mean, dtype=torch.bool),
        }
        return self._make_prediction(mean, variance, masks, query_spec.units)

    def score(self, prediction: Any, raw_query: Any) -> Any:
        """Score one prediction against the observed query through the sensor model.

        Prediction and query are paired point by point. A per-camera prediction is
        scored against that camera's slice (``raw_query.select_camera(...)``); a
        prediction is never broadcast onto another camera's query points, and a
        mismatch raises :class:`QueryAlignmentError` before the sensor model sees it.
        """
        if not hasattr(prediction, "units"):
            raise TypeError(f"prediction must carry units, got {type(prediction).__name__}")
        query = coerce_raw_query(raw_query)
        prediction_units = str(prediction.units)
        if prediction_units != query.units:
            raise ValueError(
                f"refusing to compare different units: prediction {prediction_units!r} vs "
                f"query {query.units!r}"
            )
        query = self._align_query_device(prediction, query)
        self._assert_unit_domain(prediction, query)
        self._assert_point_alignment(prediction, query)
        scorer = getattr(self.sensor_model, "score", None)
        if not callable(scorer):
            raise TypeError("sensor_model must implement score(prediction, raw_query)")
        return scorer(prediction, query)

    @staticmethod
    def _align_query_device(prediction: Any, query: RawQuery) -> RawQuery:
        """Move the query storage to the prediction's device.

        Device placement is not measurement semantics: a micro-batch step may keep
        the CPU measurement cache while the encoder/decoder run on CUDA, and the
        density must not depend on where the same declared numbers live. Values are
        copied unchanged; masks follow and ``camera_index`` follows.
        """
        mean = torch.as_tensor(prediction.mean)
        if query.values.device == mean.device:
            return query  # already aligned: no new object, no re-validation
        return RawQuery._trusted(
            values=query.values.to(mean.device),
            masks={
                str(name): value.to(mean.device) if isinstance(value, torch.Tensor) else value
                for name, value in query.masks.items()
            },
            units=query.units,
            keys=query.keys,
            camera_index=(
                None if query.camera_index is None else query.camera_index.to(mean.device)
            ),
            unit_affine=query.unit_affine,
        )

    @staticmethod
    def _assert_unit_domain(prediction: Any, query: RawQuery) -> None:
        """A declared-unit query never scores an undeclared internal-domain prediction.

        When the query owner declares that the model's internal prediction domain
        needs an affine inversion (:attr:`QuerySpec.unit_affine`), the prediction
        must declare the transform it was built with (``masks["unit_scale"]``,
        which :meth:`predict` always writes). An undeclared prediction is the
        normalized-domain/DN mismatch the unit contract exists to stop.
        """
        if not query.unit_affine:
            return
        masks = getattr(prediction, "masks", None)
        declared = masks.get("unit_scale") if isinstance(masks, Mapping) else None
        if declared is None:
            raise QueryAlignmentError(
                "the query declares an internal->declared-units affine but the prediction "
                "does not declare its unit transform (masks['unit_scale']); a prediction in "
                "the internal normalized domain must not be scored against declared-unit "
                "measurements"
            )

    @staticmethod
    def _assert_point_alignment(prediction: Any, query: RawQuery) -> None:
        """Refuse any pairing that would need broadcasting along the point axis."""
        mean = torch.as_tensor(prediction.mean)
        variance = torch.as_tensor(prediction.variance)
        values = query.values
        points = query.num_points
        if mean.shape != variance.shape:
            raise QueryAlignmentError(
                f"prediction mean {tuple(mean.shape)} and variance {tuple(variance.shape)} "
                "must share one shape"
            )
        for name, tensor in (("prediction", mean), ("query", values)):
            if tensor.dim() < 1 or int(tensor.shape[-1]) != points:
                raise QueryAlignmentError(
                    f"{name} carries {int(tensor.shape[-1]) if tensor.dim() >= 1 else 0} "
                    f"query points but the declared query has {points}; a prediction and a "
                    "query are paired per (camera, query point) and are never broadcast "
                    "across cameras"
                )
        for axis, (left, right) in enumerate(zip(mean.shape[:-1], values.shape[:-1], strict=False)):
            if left != right and left != 1 and right != 1:
                raise QueryAlignmentError(
                    f"prediction and query disagree on axis {axis} of their batch shape: "
                    f"{tuple(mean.shape[:-1])} vs {tuple(values.shape[:-1])}"
                )
        prediction_index = None
        if isinstance(getattr(prediction, "masks", None), Mapping):
            prediction_index = prediction.masks.get("camera_index")
        if prediction_index is not None and query.camera_index is not None:
            declared = _broadcast_camera_index(
                torch.as_tensor(prediction_index, dtype=torch.long), mean.shape, "prediction"
            )
            expected = _broadcast_camera_index(
                torch.as_tensor(query.camera_index, dtype=torch.long), values.shape, "query"
            )
            # non-negativity/range were validated once when the indices were built;
            # only the (pointer-fast) equality check belongs on the per-score path
            if declared.numel() and expected.numel():
                if not _indices_match(declared, expected):
                    raise QueryAlignmentError(
                        "prediction and query disagree on the camera of their query points; "
                        "score one camera at a time with RawQuery.select_camera(...)"
                    )

    # -- internals -----------------------------------------------------------------------

    def _assert_history_is_causal(self, history: HistoryContext, query_spec: QuerySpec) -> None:
        if history.last_frame_index is None:
            return  # no history: calibration-only, never an independent check
        if history.last_frame_index > query_spec.frame_index - 1:
            raise CausalLeakError(
                f"prequential history ends at frame {history.last_frame_index} but the query "
                f"is frame {query_spec.frame_index}"
            )
        query_identities: set[str] = set()
        for key in query_spec.keys:
            query_identities.update({key.source_frame_id, key.acquisition_id, key.key_hash()})
        for level_id, level in (history.levels or {}).items():
            for support in level.support_ids:
                if str(support) in query_identities:
                    raise CausalLeakError(
                        f"history level {level_id!r} carries support {support!r} that belongs "
                        f"to the current query"
                    )

    @staticmethod
    def _assert_eta_declared(history: HistoryContext) -> None:
        eta = history.eta
        if eta is None:
            return
        source = getattr(eta, "source", "history")
        if source not in ("history", "oracle"):
            raise ValueError(
                f"eta source must be 'history' or 'oracle', got {source!r}; eta estimated "
                "from anything else is not declared interference"
            )

    def _context_tensor(self, history: HistoryContext) -> tuple[torch.Tensor, bool]:
        features = history.pooled_features
        if features is None:
            if history.last_frame_index is not None:
                raise ValueError(
                    "history declares frames but carries no pooled features; refusing to "
                    "invent a context"
                )
            return torch.zeros(1, self.feature_dim), True
        tensor = _as_float_tensor(
            features, "history pooled_features", detach=False, check_finite=False
        )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"pooled_features must be (B, {self.feature_dim}), got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != self.feature_dim:
            raise ValueError(
                f"pooled_features width {tensor.shape[1]} does not match the model feature "
                f"dim {self.feature_dim}"
            )
        return tensor, history.last_frame_index is None

    def _world_vector(self, world: WorldHypothesis) -> torch.Tensor:
        cached = self._world_vectors.get(id(world))
        if cached is not None and cached[0] is world:
            return cached[1]
        vector = self._build_world_vector(world)
        if len(self._world_vectors) >= 512:
            self._world_vectors.clear()
        self._world_vectors[id(world)] = (world, vector)
        return vector

    def _build_world_vector(self, world: WorldHypothesis) -> torch.Tensor:
        positions = world_positions(world)
        if positions is None:
            raise ValueError(
                f"world {world.world_id!r} declares no view geometry; a candidate must carry "
                "its geometric hypothesis"
            )
        if positions.shape[0] != self.world_size:
            raise ValueError(
                f"world {world.world_id!r} has {positions.shape[0]} objects but the model "
                f"declares {self.world_size}"
            )
        nuisance = getattr(world, "nuisance", None)
        declared: Mapping[str, Any] = {}
        if isinstance(nuisance, Mapping):
            inner = nuisance.get("declared")
            declared = inner if isinstance(inner, Mapping) else nuisance
        _check_declared_keys(declared, f"world {world.world_id!r} nuisance")
        unknown = sorted(set(declared) - set(NUISANCE_FIELDS))
        if unknown:
            raise ValueError(
                f"world {world.world_id!r} declares unknown nuisance fields {unknown}; "
                f"declared fields are {NUISANCE_FIELDS}"
            )
        log_exposure = declared.get("log_exposure", 0.0)
        camera_shift = declared.get("camera_shift_px", (0.0, 0.0))
        noise_scale = declared.get("noise_scale", 1.0)
        # Every component is built on the geometry tensor's device/dtype: a CUDA
        # geometry must never meet a CPU nuisance scalar.
        device = positions.device
        dtype = positions.dtype

        def _scalar(value: Any) -> torch.Tensor:
            return torch.as_tensor(value, dtype=dtype, device=device)

        shift = _scalar(camera_shift).reshape(-1)
        if shift.numel() != 2:
            raise ValueError("candidate camera_shift_px must be a 2-vector")
        geometry = (
            positions.reshape(-1),
            _world_geometry_field(
                world, "scale_px", torch.ones(self.world_size, dtype=dtype, device=device)
            ),
            _world_geometry_field(
                world, "depth", torch.zeros(self.world_size, dtype=dtype, device=device)
            ),
        )
        nuisance_vector = torch.cat([_scalar([log_exposure]), shift, _scalar([noise_scale])])
        return torch.cat([*geometry, nuisance_vector])

    def _normalized_coordinates(
        self,
        query_spec: QuerySpec,
        shift: torch.Tensor | None,
        batch: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Move declared sensor-pixel coordinates into ``[-1, 1]`` with tensor ops only.

        The per-point sensor height/width comes from the spec's cached per-camera
        table, so a full-resolution query costs a few elementwise tensor operations
        instead of one Python step per point (the measurement path calls this once
        per candidate per row).
        """
        raw = query_spec.sensor_coordinates.to(
            device=device if device is not None else query_spec.sensor_coordinates.device,
            dtype=torch.float32,
        )
        if shift is None:
            raw = raw.unsqueeze(0).expand(batch, -1, -1)
        else:
            raw = raw.unsqueeze(0) - shift
        heights, widths = query_spec.coordinate_divisors(self.sensor_shapes, device=raw.device)
        result = torch.empty_like(raw)
        result[..., 0] = raw[..., 0] * 2.0 / heights - 1.0
        result[..., 1] = raw[..., 1] * 2.0 / widths - 1.0
        return result

    @staticmethod
    def _eta_translation(eta: Any, query_spec: QuerySpec, batch: int) -> torch.Tensor | None:
        """``(B, Q, 2)`` sensor-pixel camera translation declared by eta, or ``None``."""
        if eta is None:
            return None
        value = getattr(eta, "camera_translation", None)
        if value is None:
            return None
        if isinstance(value, Mapping):
            order = list(dict.fromkeys(query_spec.camera_ids))
            columns = []
            for camera_id in order:
                entry = value.get(camera_id, (0.0, 0.0))
                columns.append(torch.as_tensor(entry, dtype=torch.float32).reshape(-1, 2))
            if len({int(column.shape[0]) for column in columns}) != 1:
                raise ValueError("eta.camera_translation rows must share one batch width")
            shift = torch.stack(columns, dim=1)  # (B, V, 2)
            shift = shift[:, query_spec.camera_columns(order), :]
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
            unique_cameras = list(dict.fromkeys(query_spec.camera_ids))
            if tensor.dim() == 2 and tensor.shape[-1] == 2:
                shift = tensor.reshape(tensor.shape[0], 1, 2).expand(
                    -1, len(query_spec.camera_ids), -1
                )
            elif tensor.dim() == 3 and tensor.shape[-1] == 2:
                if tensor.shape[1] == len(unique_cameras):
                    shift = tensor[:, query_spec.camera_columns(unique_cameras), :]
                elif tensor.shape[1] == len(query_spec.camera_ids):
                    shift = tensor
                else:
                    raise ValueError(
                        f"eta.camera_translation has {tensor.shape[1]} camera columns; "
                        f"expected {len(unique_cameras)} declared cameras or "
                        f"{len(query_spec.camera_ids)} query points"
                    )
            else:
                raise ValueError(
                    "eta.camera_translation must be (B, 2), (B, V, 2) or a per-camera mapping, "
                    f"got {tuple(tensor.shape)}"
                )
            shift = shift.clone()
        if shift.shape[0] == 1 and batch > 1:
            shift = shift.expand(batch, -1, -1)
        if shift.shape[0] != batch:
            raise ValueError(
                f"eta.camera_translation batch {shift.shape[0]} does not match "
                f"context batch {batch}"
            )
        return shift

    @staticmethod
    def _apply_eta_scale(
        eta: Any, mean: torch.Tensor, variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if eta is None:
            return mean, variance
        device = mean.device
        log_exposure = getattr(eta, "log_exposure", None)
        if log_exposure is not None:
            gain = torch.exp(
                torch.as_tensor(log_exposure, dtype=torch.float32).to(device).reshape(-1, 1)
            )
            gain = gain.expand(mean.shape[0], -1) if gain.shape[0] == 1 else gain
            if gain.shape[0] != mean.shape[0]:
                raise ValueError("eta.log_exposure batch does not match the prediction batch")
            mean = mean * gain
        noise_scale = getattr(eta, "noise_scale", None)
        if noise_scale is not None:
            scale = torch.as_tensor(noise_scale, dtype=torch.float32).to(device).reshape(-1, 1)
            scale = scale.expand(variance.shape[0], -1) if scale.shape[0] == 1 else scale
            if scale.shape[0] != variance.shape[0]:
                raise ValueError("eta.noise_scale batch does not match the prediction batch")
            if bool((scale <= 0).any()):
                raise ValueError("eta.noise_scale must be positive")
            variance = variance * scale * scale
        return mean, variance

    @staticmethod
    def _apply_unit_affine(
        mean: torch.Tensor,
        variance: torch.Tensor,
        query_spec: QuerySpec,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Invert the internal prediction domain into the declared query units.

        Returns ``(mean, variance, unit_scale, unit_offset)`` where the scale and
        offset are ``(B, Q)`` float tensors recorded on the prediction, so a score
        can prove the prediction was expressed in the declared units.
        """
        device = mean.device if device is None else device
        if query_spec.affine_is_identity():
            # One cached Python bool instead of two device-wide reductions: the
            # identity case is common and must not synchronize CUDA.
            scale = torch.ones_like(mean)
            offset = torch.zeros_like(mean)
            return mean, variance, scale, offset
        scale, offset = query_spec.affine_tensors(device=device)
        scale = scale.unsqueeze(0).expand_as(mean)
        offset = offset.unsqueeze(0).expand_as(mean)
        mean = mean * scale + offset
        variance = variance * scale * scale
        return mean, variance, scale, offset

    @staticmethod
    def _eta_seen_query(history: HistoryContext, query_spec: QuerySpec) -> bool:
        if history.eta is None or not history.eta_fit_keys:
            return False
        query_identities = set()
        for key in query_spec.keys:
            query_identities.update({key.source_frame_id, key.acquisition_id, key.key_hash()})
        for key in history.eta_fit_keys:
            if key.source_frame_id in query_identities or key.acquisition_id in query_identities:
                return True
        return False

    def _raw_prediction_type(self) -> type:
        declared = getattr(self.sensor_model, "raw_prediction_type", None)
        if declared is not None:
            return declared
        from owac.frameworks.n02_full import sensor as sensor_module

        return sensor_module.RawPrediction

    def _make_prediction(
        self, mean: torch.Tensor, variance: torch.Tensor, masks: dict[str, Any], units: str
    ) -> Any:
        """Assemble the declared prediction, preferring the sensor model's own builder.

        ``SensorModel.build_prediction`` (T2) adds the declared sensor variance split
        around the decoder's model variance; a duck-typed sensor model without it gets
        the frozen ``RawPrediction`` shape directly.
        """
        builder = getattr(self.sensor_model, "build_prediction", None)
        if callable(builder):
            return builder(mean=mean, model_variance=variance, masks=masks, units=str(units))
        prediction_type = self._raw_prediction_type()
        return prediction_type(mean=mean, variance=variance, masks=masks, units=str(units))


# --------------------------------------------------------------------------------------
# analytic oracle (evaluator-only information upper bound)
# --------------------------------------------------------------------------------------


class AnalyticOracle:
    """Analytic generator oracle with the same predict/score interface.

    ``mean_variance`` is supplied by the evaluator from the declared generative
    model; the class itself never imports or reads ``EvaluationTruth`` and is never
    a member of a candidate dictionary.
    """

    def __init__(
        self,
        mean_variance: Callable[[HistoryContext, QuerySpec], tuple[Any, Any]],
        *,
        prediction_type: type,
        sensor_model: Any = None,
        version: str = "analytic-generator",
        world_id: str = "oracle",
    ) -> None:
        if not callable(mean_variance):
            raise TypeError("mean_variance must be callable(history, query_spec)")
        self.mean_variance = mean_variance
        self.prediction_type = prediction_type
        self.sensor_model = sensor_model
        self.version = str(version)
        self.world_id = str(world_id)

    def predict(
        self,
        world: WorldHypothesis | None,
        history: HistoryContext,
        query_spec: QuerySpec,
        *,
        call: str = "prequential_check",
    ) -> Any:
        if call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}, got {call!r}")
        mean, variance = self.mean_variance(history, query_spec)
        mean = _as_float_tensor(mean, "oracle mean", detach=False)
        variance = _as_float_tensor(variance, "oracle variance", detach=False)
        if mean.shape != variance.shape:
            raise ValueError(
                f"oracle mean {tuple(mean.shape)} and variance {tuple(variance.shape)} "
                "must share one shape"
            )
        if bool((variance <= 0).any()):
            raise ValueError("oracle variance must be positive")
        masks = {
            "valid": torch.ones_like(mean, dtype=torch.bool),
            "bad": torch.zeros_like(mean, dtype=torch.bool),
            "missing": torch.zeros_like(mean, dtype=torch.bool),
            "saturated": torch.zeros_like(mean, dtype=torch.bool),
            "calibration_only": torch.zeros_like(mean, dtype=torch.bool),
            "empirical_check": torch.zeros_like(mean, dtype=torch.bool),
            "oracle_eta": torch.ones_like(mean, dtype=torch.bool),
            # the oracle states its mean/variance in the declared query units already
            "unit_scale": torch.ones_like(mean, dtype=torch.float32),
            "unit_offset": torch.zeros_like(mean, dtype=torch.float32),
            "units_declared": torch.ones_like(mean, dtype=torch.bool),
        }
        return self.prediction_type(
            mean=mean, variance=variance, masks=masks, units=query_spec.units
        )

    def score(self, prediction: Any, raw_query: Any) -> Any:
        if self.sensor_model is None:
            raise RuntimeError("this analytic oracle has no sensor model to score with")
        scorer = getattr(self.sensor_model, "score", None)
        if not callable(scorer):
            raise TypeError("sensor_model must implement score(prediction, raw_query)")
        return scorer(prediction, coerce_raw_query(raw_query))

    def world(self) -> WorldHypothesis:
        entity = EntityState(entity_ids=(self.world_id,), geometry={}, geometry_frame="view_2d")
        return WorldHypothesis(
            world_id=self.world_id,
            entities=(entity,),
            nuisance=None,
            score=0.0,
            score_kind="normalized_density",
            ancestor_id=None,
            version=0,
        )


# --------------------------------------------------------------------------------------
# controlled-task assembly
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class P1Task:
    """One assembled controlled task; truth never enters the online prediction path."""

    inference: InferenceInput
    targets: TrainingTargets
    truth: Any
    dictionary: CandidateDictionary
    frame_index: int
    units: str
    candidate_index: int | None
    notes: dict = field(default_factory=dict)

    @property
    def worlds(self) -> tuple[WorldHypothesis, ...]:
        return self.dictionary.worlds()


def _first_frame_units(inference: InferenceInput) -> str:
    for frame in inference.bundle.frames.values():
        metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
        payload = metadata.get("payload") if isinstance(metadata.get("payload"), Mapping) else {}
        for source in (metadata, payload):
            for name in ("unit", "units"):
                value = source.get(name) if isinstance(source, Mapping) else None
                if value:
                    return str(value)
    return ""


def build_p1_task(
    source: Any,
    *,
    frame_index: int = 0,
    offsets: Sequence[tuple[float, float]] = DEFAULT_WRONG_OFFSETS,
    all_wrong: bool = False,
    correct_index: int = 0,
) -> P1Task:
    """Assemble the public controlled task from a generator sequence.

    ``source`` must expose ``inference`` and ``truth`` sequences (the P0
    ``SyntheticSequence`` qualifies) and may expose ``targets``. The dictionary is
    built here, from truth geometry, as the generator's *public* explanation set:
    the constructive correct index is fixed at construction and copied into
    ``TrainingTargets.optional_geometry["candidate_index"]`` for the T4 loss. On an
    all-wrong task the label key is absent (the CE term is then unavailable by
    design, never fabricated).
    """
    inference = source.inference[frame_index]
    if not isinstance(inference, InferenceInput):
        raise TypeError("source.inference must hold InferenceInput objects")
    truth = source.truth[frame_index]
    world_px = _normalise_truth_geometry(truth)
    if world_px is None:
        raise ValueError("the generator truth of this frame declares no world_px geometry")
    count = int(world_px.shape[0])
    wrong_specs: list[CandidateSpec] = []
    for index, offset in enumerate(offsets):
        positions = world_px.clone()
        positions[index % count] = positions[index % count] + torch.as_tensor(
            offset, dtype=torch.float32
        )
        if all_wrong:
            # no candidate matches the truth: perturb a second object too
            second = (index + 1) % count
            positions[second] = (
                positions[second] - torch.as_tensor(offset, dtype=torch.float32) * 0.5
            )
        wrong_specs.append(
            CandidateSpec(
                candidate_id=f"wrong-{index:02d}",
                category="wrong",
                geometry=CandidateGeometry(world_px=positions),
                nuisance={"log_exposure": 0.2 * (index + 1)},
            )
        )
    if all_wrong:
        specs = tuple(wrong_specs)
    else:
        if not 0 <= correct_index <= len(wrong_specs):
            raise ValueError(
                f"correct_index {correct_index} is outside the assembled dictionary of "
                f"{len(wrong_specs) + 1} candidates"
            )
        correct = CandidateSpec(
            candidate_id="correct",
            category="correct",
            geometry=CandidateGeometry(world_px=world_px.clone()),
        )
        specs = tuple(wrong_specs[:correct_index]) + (correct,) + tuple(wrong_specs[correct_index:])
    dictionary = CandidateDictionary(specs=specs, all_wrong=all_wrong)
    label = dictionary.label_index(truth)
    geometry_target: dict[str, Any] = {
        "world_px": world_px.clone(),
        "geometry_frame": "view_2d",
        "frame_index": int(frame_index),
    }
    if label is not None:
        geometry_target["candidate_index"] = int(label)
    source_targets = getattr(source, "targets", None)
    base_targets = source_targets[frame_index] if source_targets is not None else TrainingTargets()
    if not isinstance(base_targets, TrainingTargets):
        raise TypeError("source.targets must hold TrainingTargets objects")
    targets = TrainingTargets(
        actions=base_targets.actions,
        action_mask=base_targets.action_mask,
        optional_geometry=geometry_target,
        optional_relations=base_targets.optional_relations,
    )
    return P1Task(
        inference=inference,
        targets=targets,
        truth=truth,
        dictionary=dictionary,
        frame_index=int(frame_index),
        units=_first_frame_units(inference),
        candidate_index=label,
        notes={
            "all_wrong": bool(all_wrong),
            "categories": dictionary.categories(),
            "cross_check": dictionary.last_cross_check,
            "generator_side_only": True,
        },
    )


# --------------------------------------------------------------------------------------
# feature pooling helper
# --------------------------------------------------------------------------------------


def _grouped_feature_levels(pyramid: Any) -> dict[str, dict[str, FeatureLevel]]:
    """``{camera_id: {level_id: FeatureLevel}}`` using the declared pyramid view."""
    per_camera = getattr(pyramid, "per_camera", None)
    if callable(per_camera):
        grouped = per_camera()
        if isinstance(grouped, Mapping) and grouped:
            return {str(camera): dict(levels) for camera, levels in grouped.items()}
    levels = getattr(pyramid, "levels", None)
    if not isinstance(levels, Mapping) or not levels:
        raise TypeError("a FeaturePyramid with at least one level is required")
    cameras = tuple(str(item) for item in getattr(pyramid, "camera_ids", ()) or ())
    if len(cameras) > 1:
        raise TypeError(
            "a multi-camera FeaturePyramid must expose per_camera(); level keys are never "
            "parsed by hand"
        )
    return {cameras[0] if cameras else "": dict(levels)}


def pool_feature_pyramid(
    pyramid: Any,
    *,
    level_id: str | None = None,
    camera_id: str | None = None,
    rows: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked mean-pool one encoded feature level into a ``(B, C)`` context vector.

    The level keys are never parsed: the pyramid's ``per_camera()`` view is used
    when present, so multi-camera ``"<camera_id>::<level_id>"`` keys work unchanged
    and :attr:`FeatureLevel.level_id` stays the bare name. ``camera_id`` defaults to
    the first declared camera; a multi-camera pyramid without an explicit camera is
    still resolved to the first declared one, which the caller should override when
    a different view is the declared context.

    ``rows`` selects a subset of the encoded batch *after* the encode, so one batched
    forward can serve several decision times: each row's pooled context only ever
    contains that row's own frame, exactly as an individual encode would produce.
    """
    grouped = _grouped_feature_levels(pyramid)
    declared = tuple(str(item) for item in getattr(pyramid, "camera_ids", ()) or ())
    if camera_id is not None:
        camera = str(camera_id)
    elif declared:
        camera = declared[0]
    else:
        camera = next(iter(grouped))
    if camera not in grouped:
        raise KeyError(
            f"camera {camera!r} is not part of this pyramid; declared cameras are {sorted(grouped)}"
        )
    levels = grouped[camera]
    chosen = level_id or ("stride8" if "stride8" in levels else sorted(levels)[0])
    if chosen not in levels:
        raise KeyError(
            f"level {chosen!r} is not available for camera {camera!r}; have {sorted(levels)}"
        )
    level = levels[chosen]
    values = _as_float_tensor(
        level.values, f"level {chosen!r} values", detach=False, check_finite=False
    )
    if values.dim() != 4:
        raise ValueError(f"level {chosen!r} values must be (B, C, H, W), got {tuple(values.shape)}")
    valid = getattr(level, "valid", None)
    if rows is not None:
        positions = torch.as_tensor(rows, dtype=torch.long, device=values.device)
        if positions.dim() != 1:
            raise ValueError("rows must be one-dimensional")
        values = values.index_select(0, positions)
        if valid is not None:
            valid = torch.as_tensor(valid).index_select(0, positions)
            level = replace(level, valid=valid)
    if valid is None:
        return values.mean(dim=(2, 3))
    mask = torch.as_tensor(valid).bool()
    if mask.shape != values.shape[:1] + values.shape[2:]:
        raise ValueError(
            f"level {chosen!r} valid mask {tuple(mask.shape)} does not match "
            f"{tuple(values.shape[:1] + values.shape[2:])}"
        )
    weights = mask.unsqueeze(1).to(values.dtype)
    total = weights.sum(dim=(2, 3)).clamp_min(1.0)
    return (values * weights).sum(dim=(2, 3)) / total

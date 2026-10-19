"""Minimal causal stepping for the N02 P1 line (T3).

Frozen interface: ``docs/prompts/n02_v1_1/p1_interface_freeze.md`` sections 3.5
and 3.7. :func:`step_bundle` is the single online path shared by inference and
training; it performs, in this order:

1. per-episode reset of the recurrent state rows whose episode changed;
2. registration of every new ``MeasurementKey`` of this decision time (a repeated
   acquisition adds no evidence; a calibration change only replaces the derived
   calibration version);
3. formation of the ``prequential_check`` predictions from the declared history
   only, *before* the current target measurement is read;
4. reading of the query values and locking of the prequential checks;
5. encoding of the current RAW and scoring of the fit-mode predictions;
6. one likelihood entry per ``ObjectiveKey`` (a second pass over the same
   objective reuses the stored entry instead of accumulating another one).

Loss computation stays in the training harness (T4): this module returns raw
quantities (``locked_checks`` prequential, ``fit_scores`` fit, world scores) and
never consumes ``TrainingTargets`` for prediction or checking.

v1.2 entry (P2): the default model path of this checkout is the v1.2 chain in
:mod:`owac.frameworks.n02_full.model_v12`, exposed lazily here as ``N02V12Model`` /
``build_n02_v12_model`` / ``build_default_model``; :data:`DEFAULT_ENTRY` names it, and no default
path falls back to the synthetic generator. The P0/P1 code below is kept intact as the historical
command path. ``locked_checks`` now carries :class:`model_v12.LockedAbsoluteCheck` objects, i.e. a
real three-state verdict plus a ``valid`` flag, instead of a verdict-less ``MeasurementScore``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from owac.frameworks.n02_full.factors import (
    HistoryContext,
    ObservationModel,
    QuerySpec,
    RawQuery,
    UnitAffine,
    pool_feature_pyramid,
)
from owac.frameworks.n02_full.state import (
    FeatureLevel,
    MeasurementKey,
    MeasurementLedger,
    MeasurementRecord,
    ObjectiveKey,
    ObservationBundle,
    SequenceState,
    TrainingTargets,
    WorldHypothesis,
    canonical_json,
    measurement_set_hash,
)

__all__ = [
    "DEFAULT_ENTRY",
    "ObjectiveRegister",
    "RowQuery",
    "StepOutput",
    "build_default_model",
    "build_raw_query",
    "build_row_query",
    "step_bundle",
]

# The v1.2 chain is the default model path; the P0/P1 line stays reachable as history.
DEFAULT_ENTRY = "n02_v12"
_V12_EXPORTS = ("N02V12Model", "build_n02_v12_model", "build_default_v12_model")


def __getattr__(name: str) -> Any:
    """Expose the v1.2 entry lazily so importing ``model`` stays cheap and P0/P1-safe."""
    if name in _V12_EXPORTS:
        from owac.frameworks.n02_full import model_v12

        return getattr(model_v12, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_default_model(config: Any = None, **overrides: Any) -> Any:
    """Build the default model of this checkout: the v1.2 chain, never the synthetic path."""
    from owac.frameworks.n02_full import model_v12

    return model_v12.build_n02_v12_model(config, **overrides)


SCORE_KIND_OF = {
    "proper_nll": "normalized_density",
    "empirical_energy": "empirical_energy",
    "importance_weight": "importance_weight",
}


@dataclass(slots=True)
class StepOutput:
    """Raw output of one causal step; losses are computed by the training harness.

    ``world_scores`` is the differentiable view of the fit objective: one entry
    per world of ``worlds``, in the same order, computed from the same explicit
    measurement set as ``fit_scores``. ``worlds[k].score`` is the detached Python
    float readout of exactly that quantity (``worlds[k].score_kind`` declares how
    it may be read); the candidate cross-entropy of the training harness consumes
    ``world_scores`` because a Python float carries no gradient.
    """

    objective: ObjectiveKey
    worlds: tuple[WorldHypothesis, ...] = ()
    best_world: WorldHypothesis | None = None
    locked_checks: tuple[Any, ...] = ()
    fit_scores: tuple[Any, ...] = ()
    world_scores: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    state: SequenceState | None = None
    ledger_version: int = 0
    evidence_count: int = 0
    residuals: dict[str, float] = field(default_factory=dict)
    cost: dict[str, float] = field(default_factory=dict)


class ObjectiveRegister:
    """One likelihood entry per ``ObjectiveKey``; re-runs reuse the first entry.

    The objective identity is the frozen tuple (measurement set, model version,
    structure version, calibration versions, initial prior version). Evaluating K
    candidates or replaying the same objective under one identity never adds a
    second likelihood term.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, tuple[ObjectiveKey, float]] = {}

    def add(self, key: ObjectiveKey, value: float) -> tuple[bool, float]:
        """Register ``value``; returns ``(is_new, stored_value)``."""
        if not isinstance(key, ObjectiveKey):
            raise TypeError(f"key must be an ObjectiveKey, got {type(key).__name__}")
        digest = key.key_hash()
        if digest in self._entries:
            return False, self._entries[digest][1]
        stored = float(value)
        self._entries[digest] = (key, stored)
        return True, stored

    def value(self, key: ObjectiveKey) -> float | None:
        entry = self._entries.get(key.key_hash())
        return None if entry is None else entry[1]

    def keys(self) -> tuple[ObjectiveKey, ...]:
        return tuple(key for key, _ in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, ObjectiveKey) and key.key_hash() in self._entries


# --------------------------------------------------------------------------------------
# frame-level helpers (duck-typed over the shared MeasurementFrame contract)
# --------------------------------------------------------------------------------------


def _dig(mapping: Any, path: Sequence[str]) -> Any:
    value = mapping
    for step in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(step)
    return value


def _first_declared(frame: Any, paths: Sequence[Sequence[str]]) -> Any:
    metadata = frame.metadata if isinstance(frame.metadata, Mapping) else {}
    for path in paths:
        value = _dig(metadata, path)
        if value is not None:
            return value
    return None


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return (len(value),)
    return None


def _is_batched_frame(frame: Any, batch: int) -> bool:
    """Whether this frame carries one row per batch element."""
    if batch <= 1:
        return False
    if len(tuple(frame.episode_ids)) == batch:
        return True
    shape = _shape_of(frame.values)
    return bool(shape is not None and len(shape) >= 3 and shape[0] == batch)


def _take_row(value: Any, row: int, batch: int, *, batched: bool, name: str) -> Any:
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if not batched:
        return tensor
    if tensor.dim() < 1 or tensor.shape[0] != batch:
        raise ValueError(f"{name} has shape {tuple(tensor.shape)} but the bundle batch is {batch}")
    return tensor[row]


def _frame_payload(frame: Any, row: int, batch: int) -> dict[str, torch.Tensor | None]:
    batched = _is_batched_frame(frame, batch)
    return {
        "values": _take_row(frame.values, row, batch, batched=batched, name="frame values"),
        "valid": _take_row(frame.valid_mask, row, batch, batched=batched, name="valid_mask"),
        "bad": _take_row(frame.bad_mask, row, batch, batched=batched, name="bad_mask"),
        "saturated": _take_row(
            frame.saturated_mask, row, batch, batched=batched, name="saturated_mask"
        ),
    }


def _frame_id(frame: Any, row: int, batch: int, camera_id: str) -> str:
    identifiers = tuple(str(item) for item in frame.frame_ids)
    if identifiers:
        if _is_batched_frame(frame, batch) and len(identifiers) == batch:
            return identifiers[row]
        return identifiers[0]
    return f"{camera_id}:{row}"


def _declared_acquisition_id(frame: Any) -> str | None:
    value = _first_declared(
        frame,
        (
            ("acquisition_id",),
            ("payload", "acquisition_id"),
            ("camera", "acquisition_id"),
        ),
    )
    return None if value is None else str(value)


def _measurement_key(
    frame: Any, *, row: int, batch: int, episode_id: str, camera_id: str
) -> MeasurementKey:
    source_frame_id = _frame_id(frame, row, batch, camera_id)
    declared = _declared_acquisition_id(frame)
    acquisition_id = f"{declared}:{source_frame_id}" if declared else source_frame_id
    return MeasurementKey(
        episode_id=episode_id,
        camera_id=camera_id,
        source_frame_id=source_frame_id,
        acquisition_id=acquisition_id,
    )


def _calibration_version(frame: Any) -> str:
    declared = _first_declared(
        frame,
        (
            ("calibration_version",),
            ("payload", "calibration_version"),
        ),
    )
    if declared:
        return str(declared)
    calibration = _first_declared(
        frame,
        (
            ("calibration",),
            ("payload", "calibration"),
        ),
    )
    if isinstance(calibration, Mapping):
        payload = dict(calibration)
        try:
            text = canonical_json(payload)
        except TypeError:
            text = str(payload)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return ""


def _raw_ref(frame: Any, row: int, batch: int, camera_id: str) -> str:
    declared = _first_declared(frame, (("raw_ref",), ("payload", "raw_ref")))
    if declared:
        return str(declared)
    return _frame_id(frame, row, batch, camera_id)


def _frame_units(frame: Any) -> str:
    declared = _first_declared(
        frame,
        (
            ("unit",),
            ("units",),
            ("payload", "unit"),
            ("payload", "units"),
        ),
    )
    if not declared:
        raise ValueError(
            "the measurement frame declares no unit; a score cannot be compared without one"
        )
    return str(declared)


def _crop_offset(frame: Any) -> tuple[float, float]:
    declared = _first_declared(
        frame,
        (
            ("camera", "crop_offset"),
            ("payload", "camera", "crop_offset"),
            ("crop_offset",),
            ("payload", "crop_offset"),
        ),
    )
    if declared is not None and len(tuple(declared)) == 2:
        return float(tuple(declared)[0]), float(tuple(declared)[1])
    return 0.0, 0.0


def _query_indices(height: int, width: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    if height < 1 or width < 1:
        raise ValueError(f"a query grid needs a positive spatial extent, got {(height, width)}")
    if stride < 1:
        raise ValueError(f"query stride must be positive, got {stride}")
    rows = torch.arange(0, height, stride)
    columns = torch.arange(0, width, stride)
    grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
    return grid_y.reshape(-1), grid_x.reshape(-1)


@dataclass(slots=True)
class RowQuery:
    """One decision time's query: the :class:`QuerySpec` plus how to sample it.

    Returned by :meth:`ObservationModel.build_row_query`; pass it to
    :meth:`ObservationModel.build_raw_query` to obtain the matching
    :class:`RawQuery`. ``bundle``/``row``/``batch`` record the exact inputs, so the
    evaluator can rebuild the pair the training path used without guessing the
    sampling configuration.

    Fields an evaluator needs:

    * ``spec.keys`` -- the measurement identity of every camera of this decision time;
    * ``spec.sensor_coordinates`` -- ``(Q, 2)`` sensor-pixel coordinates of the samples;
    * ``spec.camera_ids`` -- the camera of each of the ``Q`` points;
    * ``spec.units`` -- the declared measurement unit every score must match verbatim;
    * ``spec.unit_affine`` -- internal->declared-units affine per camera (None when the
      internal prediction domain already is the declared unit).
    """

    spec: QuerySpec
    sampling: dict[str, tuple[Any, torch.Tensor, torch.Tensor]]
    bundle: ObservationBundle | None = None
    row: int = 0
    batch: int = 1


def _frame_calibration(frame: Any) -> Mapping | None:
    value = _first_declared(frame, (("calibration",), ("payload", "calibration")))
    return value if isinstance(value, Mapping) else None


def _config_affine(config: Any, camera_id: str) -> Mapping | None:
    """Config-declared internal->units affine for cameras without frame calibration."""
    declared = getattr(config, "unit_affine", None) if config is not None else None
    if isinstance(declared, Mapping):
        if camera_id in declared and isinstance(declared[camera_id], Mapping):
            return declared[camera_id]
        if {"scale", "offset"} & set(declared):
            return declared
    fallback = getattr(config, "unit_affine_default", None) if config is not None else None
    return fallback if isinstance(fallback, Mapping) else None


def _frame_unit_affine(
    frame: Any, *, units: str, prediction_domain: str, config: Any = None, camera_id: str = ""
) -> UnitAffine | None:
    """The declared affine that inverts the model's internal domain into ``units``.

    ``prediction_domain="units"`` declares that the decoder already emits declared
    units (identity). ``"normalized"`` (the default, matching T1's declared input
    normalization) inverts the frame's ``calibration`` affine ``(black, white)``;
    an explicit ``config.unit_affine`` fallback covers caches that declare no
    calibration, and a frame without either stays in declared units (no transform).
    """
    version = _calibration_version(frame)
    if prediction_domain not in ("normalized", "internal"):
        return UnitAffine(scale=1.0, offset=0.0, units=units, calibration_version=version)
    calibration = _frame_calibration(frame)
    if calibration is not None:
        black = calibration.get("black_level")
        white = calibration.get("white_level")
        if black is not None and white is not None:
            black, white = float(black), float(white)
            if white > black:
                return UnitAffine(
                    scale=white - black,
                    offset=black,
                    units=units,
                    calibration_version=version,
                )
    fallback = _config_affine(config, camera_id)
    if fallback is not None and "scale" in fallback:
        return UnitAffine(
            scale=float(fallback["scale"]),
            offset=float(fallback.get("offset", 0.0)),
            units=units,
            calibration_version=version,
        )
    return None


def _build_row_query_internal(
    bundle: ObservationBundle,
    *,
    row: int,
    batch: int,
    episode_id: str,
    history: HistoryContext,
    stride: int,
    prediction_domain: str = "normalized",
    config: Any = None,
) -> RowQuery:
    keys: list[MeasurementKey] = []
    coordinates: list[torch.Tensor] = []
    camera_ids: list[str] = []
    sampling: dict[str, tuple[Any, torch.Tensor, torch.Tensor]] = {}
    units: set[str] = set()
    affines: dict[str, UnitAffine] = {}
    first_frame_id = ""
    for camera_id, frame in bundle.frames.items():
        payload = _frame_payload(frame, row, batch)
        values = payload["values"]
        if values is None:
            raise ValueError(f"camera {camera_id!r} carries no values")
        if values.dim() == 2:
            values = values.unsqueeze(0)
        if values.dim() != 3:
            raise ValueError(
                f"camera {camera_id!r} values must be [C, H, W] after row selection, got "
                f"{tuple(values.shape)}"
            )
        height, width = int(values.shape[1]), int(values.shape[2])
        grid_y, grid_x = _query_indices(height, width, stride)
        offset_y, offset_x = _crop_offset(frame)
        pixels = torch.stack(
            [
                grid_y.to(torch.float32) + 0.5 + offset_y,
                grid_x.to(torch.float32) + 0.5 + offset_x,
            ],
            dim=-1,
        )
        channels = int(values.shape[0])
        keys.append(
            _measurement_key(
                frame, row=row, batch=batch, episode_id=episode_id, camera_id=camera_id
            )
        )
        coordinates.append(pixels.repeat(channels, 1))
        camera_ids.extend([camera_id] * (channels * int(pixels.shape[0])))
        sampling[camera_id] = (frame, grid_y, grid_x)
        frame_units = _frame_units(frame)
        units.add(frame_units)
        affine = _frame_unit_affine(
            frame,
            units=frame_units,
            prediction_domain=prediction_domain,
            config=config,
            camera_id=camera_id,
        )
        if affine is not None and not affine.is_identity:
            affines[camera_id] = affine
        if not first_frame_id:
            first_frame_id = _frame_id(frame, row, batch, camera_id)
    if not keys:
        raise ValueError("the bundle carries no camera frames to query")
    if len(units) != 1:
        raise ValueError(
            f"one decision time mixes measurement units {sorted(units)}; the P1 query "
            "spec declares a single unit"
        )
    matched = re.search(r"(\d+)(?!.*\d)", first_frame_id)
    if matched:
        frame_index = int(matched.group(1))
    elif history.last_frame_index is not None:
        frame_index = history.last_frame_index + 1
    else:
        frame_index = 0
    spec = QuerySpec._trusted(
        keys=tuple(keys),
        sensor_coordinates=torch.cat(coordinates, dim=0),
        camera_ids=tuple(camera_ids),
        frame_index=frame_index,
        units=units.pop(),
        unit_affine=affines or None,
    )
    return RowQuery(spec=spec, sampling=sampling, bundle=bundle, row=row, batch=batch)


#: Backwards-compatible private aliases: the internal builders are the public ones.
_build_row_query = _build_row_query_internal
_row_query_internal = _build_row_query_internal


def _build_raw_query(
    bundle: ObservationBundle, row_query: RowQuery, *, row: int, batch: int
) -> RawQuery:
    values: list[torch.Tensor] = []
    valid: list[torch.Tensor] = []
    bad: list[torch.Tensor] = []
    saturated: list[torch.Tensor] = []
    camera_index: torch.Tensor | None = None
    for camera_id, (frame, grid_y, grid_x) in row_query.sampling.items():
        payload = _frame_payload(frame, row=row, batch=batch)
        if payload.get("values") is None:
            raise ValueError(f"camera {camera_id!r} carries no values")
        raw = payload["values"]
        if raw.dim() == 2:
            raw = raw.unsqueeze(0)
        # the declared units are a float scale (uint16 DN included): index in float32
        # and keep the indices on the measurement's own device.
        raw = raw.to(torch.float32)
        local_y = grid_y.to(raw.device)
        local_x = grid_x.to(raw.device)
        sampled = raw[:, local_y, local_x].reshape(1, -1)
        values.append(sampled)
        shape = raw.shape[1:]
        for target, name in ((valid, "valid"), (bad, "bad"), (saturated, "saturated")):
            mask = payload[name]
            if mask is None:
                mask = (
                    torch.ones(shape, dtype=torch.bool)
                    if name == "valid"
                    else torch.zeros(shape, dtype=torch.bool)
                )
            mask = torch.as_tensor(mask).bool()
            if mask.device != raw.device:
                mask = mask.to(raw.device)
            target.append(mask[local_y, local_x].reshape(1, -1))
        if camera_index is None:
            # (camera, query-point) pairing: the spec owns the per-point index, shared
            # with the prediction's mask (identity is then proven without a device
            # comparison or a CUDA synchronization).
            camera_index = row_query.spec.camera_index(sampled.device).reshape(1, -1)
    valid_all = torch.cat(valid, dim=1)
    bad_all = torch.cat(bad, dim=1)
    missing_all = (~valid_all) & (~bad_all)
    return RawQuery._trusted(
        values=torch.cat(values, dim=1),
        masks={
            "valid": valid_all,
            "bad": bad_all,
            "missing": missing_all,
            "saturated": torch.cat(saturated, dim=1),
        },
        units=row_query.spec.units,
        keys=row_query.spec.keys,
        camera_index=camera_index,
        unit_affine=row_query.spec.unit_affine,
    )


# --------------------------------------------------------------------------------------
# public query construction (evaluator-facing; the training path uses the same code)
# --------------------------------------------------------------------------------------


def build_row_query(
    model: ObservationModel,
    bundle: ObservationBundle,
    row: int = 0,
    *,
    batch: int | None = None,
    history: HistoryContext | None = None,
    episode_id: str | None = None,
    stride: int | None = None,
) -> RowQuery:
    """Build the query of one batch row exactly as the training path does.

    Inputs: the same ``bundle``, ``row`` and sampling configuration ``step_bundle``
    uses (``stride``/``prediction_domain`` default to the model config, the episode
    id defaults to the bundle's, and ``history`` is only needed when a frame id
    carries no parseable index). Returns the :class:`RowQuery` that
    :func:`build_raw_query` turns into the observed values.

    Evaluator-relevant :class:`QuerySpec` fields: ``keys`` (measurement identity per
    camera), ``sensor_coordinates`` (``(Q, 2)`` sensor pixels), ``camera_ids``
    (camera of each point), ``units`` (declared unit string) and ``unit_affine``
    (internal->declared-units affine per camera; None when predictions already are
    declared in units).
    """
    episode_ids = _bundle_episode_ids(bundle)
    resolved_batch = len(episode_ids) if batch is None else int(batch)
    if resolved_batch < 1:
        resolved_batch = 1
    if row < 0 or row >= resolved_batch:
        raise IndexError(f"row {row} is outside the bundle batch {resolved_batch}")
    resolved_episode = (
        episode_id
        if episode_id is not None
        else (episode_ids[row] if row < len(episode_ids) else "")
    )
    resolved_history = (
        history
        if history is not None
        else HistoryContext(
            pooled_features=torch.zeros(1, int(model.feature_dim)), last_frame_index=None
        )
    )
    config = model.config
    resolved_stride = int(stride) if stride is not None else int(getattr(config, "query_stride", 8))
    return _build_row_query_internal(
        bundle,
        row=int(row),
        batch=resolved_batch,
        episode_id=str(resolved_episode),
        history=resolved_history,
        stride=resolved_stride,
        prediction_domain=str(getattr(config, "prediction_domain", "normalized")),
        config=config,
    )


def build_raw_query(
    row_query: RowQuery,
    *,
    bundle: ObservationBundle | None = None,
    row: int | None = None,
    batch: int | None = None,
) -> RawQuery:
    """Observed values/masks of a :func:`build_row_query` result.

    Inputs default to the ``(bundle, row, batch)`` recorded on ``row_query``, so
    ``build_raw_query(build_row_query(model, bundle, row))`` reproduces the exact
    :class:`RawQuery` the training path scores: ``values``, the ``valid``/``bad``/
    ``missing``/``saturated`` masks, ``units``, ``keys`` and the per-point
    ``camera_index``.
    """
    if not isinstance(row_query, RowQuery):
        raise TypeError(f"row_query must be a RowQuery, got {type(row_query).__name__}")
    source = bundle if bundle is not None else row_query.bundle
    if source is None:
        raise ValueError("this RowQuery does not record its bundle; pass bundle= explicitly")
    return _build_raw_query(
        source,
        row_query,
        row=row_query.row if row is None else int(row),
        batch=row_query.batch if batch is None else int(batch),
    )


# --------------------------------------------------------------------------------------
# batch-level helpers
# --------------------------------------------------------------------------------------


def _bundle_episode_ids(bundle: ObservationBundle) -> tuple[str, ...]:
    if bundle.episode_ids:
        return tuple(str(item) for item in bundle.episode_ids)
    for frame in bundle.frames.values():
        identifiers = tuple(str(item) for item in frame.episode_ids)
        if identifiers:
            return identifiers
    return ("",)


def _new_time_flags(bundle: ObservationBundle, batch: int) -> list[bool]:
    mask = bundle.new_time_mask
    if mask is not None:
        values = mask.tolist() if hasattr(mask, "tolist") else list(mask)
        flags = [bool(item) for item in values]
        if len(flags) != batch:
            raise ValueError(
                f"new_time_mask has {len(flags)} entries but the bundle batch is {batch}"
            )
        return flags
    declared: list[bool] | None = None
    for frame in bundle.frames.values():
        if frame.new_measurement is None:
            continue
        value = frame.new_measurement
        items = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(items, (list, tuple)):
            declared = [bool(item) for item in items]
        else:
            declared = [bool(items)]
        break
    if declared is None:
        # A missing flag means "must not assume new" (MeasurementFrame contract).
        return [False] * batch
    if len(declared) == batch:
        return declared
    if len(declared) == 1:
        return [declared[0]] * batch
    raise ValueError(
        f"frame new_measurement declares {len(declared)} entries but the bundle batch is {batch}"
    )


def _select_pooled(value: Any, row: int, batch: int) -> Any:
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    if tensor.shape[0] == batch:
        return tensor[row : row + 1]
    if tensor.shape[0] == 1:
        return tensor
    raise ValueError(
        f"history pooled features have batch {tensor.shape[0]} but the bundle batch is {batch}"
    )


def _select_level(level: FeatureLevel, row: int, batch: int) -> FeatureLevel:
    def take(value: Any) -> Any:
        if value is None:
            return None
        tensor = torch.as_tensor(value)
        if batch > 1 and tensor.dim() >= 1 and tensor.shape[0] == batch:
            return tensor[row]
        return tensor

    return FeatureLevel(
        values=take(level.values),
        valid=take(level.valid),
        sensor_coordinates=take(level.sensor_coordinates),
        support_ids=level.support_ids,
        level_id=level.level_id,
    )


def _select_eta(eta: Any, row: int, batch: int) -> Any:
    if eta is None or batch <= 1:
        return eta
    updates: dict[str, Any] = {}
    for name in ("log_exposure", "camera_translation", "noise_scale"):
        value = getattr(eta, name, None)
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch:
            updates[name] = value[row : row + 1]
    if not updates:
        return eta
    try:
        return dataclasses.replace(eta, **updates)
    except (TypeError, ValueError):
        return eta


def _select_ids(identifiers: Sequence[str], row: int, batch: int) -> tuple[str, ...]:
    values = tuple(str(item) for item in identifiers)
    if len(values) == batch:
        return (values[row],)
    return values


def _row_history(history: HistoryContext, row: int, batch: int) -> HistoryContext:
    levels = None
    if history.levels:
        levels = {
            level_id: _select_level(level, row, batch) for level_id, level in history.levels.items()
        }
    return HistoryContext(
        pooled_features=_select_pooled(history.pooled_features, row, batch),
        levels=levels,
        episode_ids=_select_ids(history.episode_ids, row, batch),
        anchor_ids=_select_ids(history.anchor_ids, row, batch),
        last_frame_index=history.last_frame_index,
        eta=_select_eta(history.eta, row, batch),
        eta_fit_keys=history.eta_fit_keys,
    )


def _slice_feature_levels(levels: Any, rows: Sequence[int]) -> Any:
    """Row-slice every level on values/valid/sensor_coordinates; ids are preserved."""
    if levels is None:
        return None
    if not isinstance(levels, Mapping):
        raise TypeError(f"levels must be a mapping, got {type(levels).__name__}")
    positions = torch.as_tensor(list(rows), dtype=torch.long)
    sliced: dict[str, Any] = {}
    for key, level in levels.items():

        def take(value: Any, name: str = str(key)) -> Any:
            if value is None:
                return None
            tensor = torch.as_tensor(value)
            if tensor.dim() < 1:
                raise ValueError(f"level {name!r} field has no batch axis: {tuple(tensor.shape)}")
            return tensor.index_select(0, positions.to(tensor.device))

        sliced[str(key)] = dataclasses.replace(
            level,
            values=take(level.values),
            valid=take(level.valid),
            sensor_coordinates=take(level.sensor_coordinates),
        )
    return sliced


def _select_bundle_rows(bundle: ObservationBundle, rows: Sequence[int]) -> Any:
    """One bundle holding only ``rows``, or ``None`` when the frames are not batched."""
    count = len(bundle.episode_ids)
    positions = torch.as_tensor(list(rows), dtype=torch.long)
    frames: dict[str, Any] = {}
    for camera_id, frame in bundle.frames.items():
        if not _is_batched_frame(frame, count):
            return None

        def take(value: Any) -> Any:
            if value is None:
                return None
            return torch.as_tensor(value).index_select(0, positions)

        def take_ids(values: Any) -> tuple:
            items = tuple(values)
            if len(items) == count:
                return tuple(items[int(index)] for index in rows)
            return items

        frames[camera_id] = dataclasses.replace(
            frame,
            values=take(frame.values),
            valid_mask=take(frame.valid_mask),
            bad_mask=take(frame.bad_mask),
            saturated_mask=take(frame.saturated_mask),
            episode_ids=take_ids(frame.episode_ids),
            frame_ids=take_ids(frame.frame_ids),
            new_measurement=take(frame.new_measurement),
        )
    return ObservationBundle(
        frames=frames,
        context=bundle.context,
        episode_ids=tuple(bundle.episode_ids[int(index)] for index in rows),
        anchor_ids=(
            tuple(bundle.anchor_ids[int(index)] for index in rows)
            if len(bundle.anchor_ids) == count
            else bundle.anchor_ids
        ),
        new_time_mask=torch.ones(len(rows), dtype=torch.bool),
    )


def _concat_predictions(predictions: Sequence[Any]) -> Any:
    """Concatenate per-row predictions of one world along the batch axis."""
    if len(predictions) == 1:
        return predictions[0]
    first = predictions[0]
    mean = torch.cat([torch.as_tensor(item.mean) for item in predictions], dim=0)
    variance = torch.cat([torch.as_tensor(item.variance) for item in predictions], dim=0)
    shared = set(predictions[0].masks)
    for item in predictions[1:]:
        shared &= set(item.masks)
    masks = {
        name: torch.cat([torch.as_tensor(item.masks[name]) for item in predictions], dim=0)
        for name in sorted(shared)
    }
    return type(first)(mean=mean, variance=variance, masks=masks, units=first.units)


def _concat_raw_queries(queries: Sequence[RawQuery]) -> RawQuery:
    if len(queries) == 1:
        return queries[0]
    names = set(queries[0].masks)
    for query in queries[1:]:
        names &= set(query.masks)
    camera_index = None
    if all(query.camera_index is not None for query in queries):
        camera_index = torch.cat(
            [
                torch.as_tensor(query.camera_index, dtype=torch.long).reshape(1, -1)
                for query in queries
            ],
            dim=0,
        )
    return RawQuery._trusted(
        values=torch.cat([query.values for query in queries], dim=0),
        masks={
            name: torch.cat([torch.as_tensor(query.masks[name]) for query in queries], dim=0)
            for name in sorted(names)
        },
        units=queries[0].units,
        keys=queries[0].keys,
        camera_index=camera_index,
        unit_affine=queries[0].unit_affine,
    )


def _mark_check(score: Any, *, calibration_only: bool, empirical: bool) -> Any:
    """Mark a check: a calibration-only or eta-fitted check is not a proper check."""
    stats = getattr(score, "local_stats", None)
    if not isinstance(stats, dict):
        raise TypeError("MeasurementScore.local_stats must be a dict")
    stats["calibration_only"] = torch.tensor(bool(calibration_only))
    stats["empirical_check"] = torch.tensor(bool(empirical))
    if calibration_only or empirical:
        try:
            return dataclasses.replace(score, kind="empirical_energy")
        except (TypeError, ValueError):
            try:
                score.kind = "empirical_energy"
            except (AttributeError, dataclasses.FrozenInstanceError):
                raise
    return score


def _absolute_locked_check(
    score: Any,
    *,
    prediction: Any,
    raw: Any,
    model: Any,
    reason: str | None = None,
) -> Any:
    """Wrap one prequential score in a real three-state check (P1 defect A).

    A verdict-less score is what made every locked check read as the string ``None`` and turned two
    broken counters into "0 false alarms / 0 detections". This returns a
    :class:`model_v12.LockedAbsoluteCheck`: the same score fields the P1 consumers already read,
    plus ``status``/``valid``/``verdict``. Whenever the check cannot be computed it is marked
    ``valid=False`` with a machine-readable reason, so statistics count it separately instead of
    scoring it as 0.
    """
    from owac.frameworks.n02_full import model_v12

    residuals = model_v12.prediction_residuals(prediction, raw)
    if reason is None and residuals is None:
        reason = "residual field unavailable for this prediction"
    return model_v12.locked_absolute_check(
        score=score,
        residuals=residuals,
        masks=getattr(prediction, "masks", None),
        calibrator=getattr(model, "calibrator", None),
        calibration_version=str(getattr(model, "calibration_version", "uncalibrated")),
        reason=reason,
    )


def _score_scalar_tensor(score: Any) -> torch.Tensor:
    """The differentiable scalar of one measurement score (a proper score is a tensor)."""
    value = torch.as_tensor(getattr(score, "raw_nll_or_energy", None))
    if value.numel() == 0:
        raise ValueError("a measurement score carries no scalar entries")
    return value.reshape(-1).mean()


def _score_mean(score: Any) -> float:
    return float(_score_scalar_tensor(score).detach())


def _world_score_kind(score_kind: str) -> str:
    if score_kind not in SCORE_KIND_OF:
        raise ValueError(
            f"unknown score kind {score_kind!r}; declared kinds are {tuple(sorted(SCORE_KIND_OF))}"
        )
    return SCORE_KIND_OF[score_kind]


def _objective_register(model: Any) -> ObjectiveRegister:
    register = getattr(model, "objective_register", None)
    if register is None:
        register = ObjectiveRegister()
        model.objective_register = register
    return register


def _objective_key(ledger: MeasurementLedger, model: Any, episode_id: str) -> ObjectiveKey:
    records = [record for record in ledger.records() if record.key.episode_id == episode_id]
    calibrations = tuple(sorted({record.calibration_version for record in records}))
    return ObjectiveKey(
        measurement_set_hash=measurement_set_hash(record.key for record in records),
        model_version=str(getattr(model, "version", "")),
        structure_version=str(getattr(model, "structure_version", "p1-fixed-entities")),
        calibration_versions=calibrations,
        initial_prior_version=str(getattr(model, "initial_prior_version", "")),
    )


def _assert_no_oracle_eta(history: HistoryContext) -> None:
    eta = history.eta
    if eta is not None and getattr(eta, "source", "history") == "oracle":
        raise ValueError(
            "oracle eta cannot enter the online step_bundle path; the oracle interference "
            "estimate is evaluator-only, while the online path consumes eta estimated from "
            "history"
        )


# --------------------------------------------------------------------------------------
# the causal step
# --------------------------------------------------------------------------------------


def step_bundle(
    bundle: ObservationBundle,
    state: SequenceState,
    model: ObservationModel,
    history: HistoryContext,
    *,
    ledger: MeasurementLedger | None = None,
    budget: Any = None,
    targets: TrainingTargets | None = None,
    fit_pyramid: Any = None,
    fit_rows: Sequence[int] | None = None,
    fit_levels: Any = None,
) -> StepOutput:
    """One minimal causal step of the P1 line (see the module docstring)."""
    if not isinstance(bundle, ObservationBundle):
        raise TypeError(f"bundle must be an ObservationBundle, got {type(bundle).__name__}")
    if not isinstance(state, SequenceState):
        raise TypeError(f"state must be a SequenceState, got {type(state).__name__}")
    if not isinstance(history, HistoryContext):
        raise TypeError(f"history must be a HistoryContext, got {type(history).__name__}")
    if not callable(getattr(model, "predict", None)) or not callable(getattr(model, "score", None)):
        raise TypeError(
            "model must provide predict(world, history, query_spec, call=...) and score(...)"
        )
    if targets is not None and not isinstance(targets, TrainingTargets):
        raise TypeError(f"targets must be a TrainingTargets, got {type(targets).__name__}")
    if budget is not None and int(getattr(budget, "max_updates", 1)) < 1:
        raise ValueError(
            "budget.max_updates must allow the single P1 update cycle, got "
            f"{getattr(budget, 'max_updates', None)!r}"
        )
    _assert_no_oracle_eta(history)
    register = _objective_register(model)
    ledger = MeasurementLedger() if ledger is None else ledger
    if not isinstance(ledger, MeasurementLedger):
        raise TypeError(f"ledger must be a MeasurementLedger, got {type(ledger).__name__}")

    episode_ids = _bundle_episode_ids(bundle)
    batch = len(episode_ids)
    flags = _new_time_flags(bundle, batch)

    # 1. per-episode reset: only the rows whose episode changed are cleared.
    previous = tuple(state.episode_ids)
    reset_rows = [
        row
        for row in range(min(batch, len(previous)))
        if previous[row] and previous[row] != episode_ids[row]
    ]
    if reset_rows:
        if state.batch_size is not None and state.batch_size != batch:
            raise ValueError(
                f"state batch {state.batch_size} does not match the bundle batch {batch}"
            )
        reset_mask = [row in set(reset_rows) for row in range(batch)]
        state = state.reset_batch(reset_mask)
    state = SequenceState(
        episode_ids=episode_ids,
        slot_hidden=state.slot_hidden,
        existence=state.existence,
        geometry=state.geometry,
        version=state.version,
    )

    # 2. register every new MeasurementKey of this decision time.
    new_evidence = 0
    for row in range(batch):
        for camera_id, frame in bundle.frames.items():
            key = _measurement_key(
                frame,
                row=row,
                batch=batch,
                episode_id=episode_ids[row],
                camera_id=camera_id,
            )
            calibration = _calibration_version(frame)
            if ledger.has(key):
                ledger.replace_calibration(key, calibration)
                continue
            record = MeasurementRecord(
                key=key,
                raw_ref=_raw_ref(frame, row, batch, camera_id),
                calibration_version=calibration,
            )
            if ledger.register(record):
                new_evidence += 1

    worlds = tuple(model.candidates.worlds()) if getattr(model, "candidates", None) else ()
    new_rows = [row for row in range(batch) if flags[row]]
    config = getattr(model, "config", None)
    stride = int(getattr(config, "query_stride", 8)) if config is not None else 8
    prediction_domain = str(getattr(config, "prediction_domain", "normalized"))

    # 3. prequential predictions from the declared history, before reading the target.
    row_queries: dict[int, RowQuery] = {}
    prequential: dict[int, list[Any]] = {}
    for row in new_rows:
        row_query = _build_row_query_internal(
            bundle,
            row=row,
            batch=batch,
            episode_id=episode_ids[row],
            history=history,
            stride=stride,
            prediction_domain=prediction_domain,
            config=config,
        )
        row_queries[row] = row_query
        if worlds:
            row_history = _row_history(history, row, batch)
            prequential[row] = [
                model.predict(world, row_history, row_query.spec, call="prequential_check")
                for world in worlds
            ]
    if len(new_rows) > 1:
        point_counts = {row_queries[row].spec.num_points for row in new_rows}
        if len(point_counts) != 1:
            raise ValueError(
                f"new-time rows declare different query sizes {sorted(point_counts)}; one "
                "bundled step needs one shared query grid"
            )
    locked_predictions: list[Any] = []
    if worlds and new_rows:
        for index in range(len(worlds)):
            locked_predictions.append(
                _concat_predictions([prequential[row][index] for row in new_rows])
            )

    # 4. read the query values and lock the prequential checks.
    raw_queries: dict[int, RawQuery] = {
        row: _build_raw_query(bundle, row_queries[row], row=row, batch=batch) for row in new_rows
    }
    locked_checks: list[Any] = []
    calibration_only_count = 0
    empirical_count = 0
    if worlds and new_rows:
        combined_raw = _concat_raw_queries([raw_queries[row] for row in new_rows])
        # the flags are uniform across candidates and rows: read them once per step
        # instead of one device synchronization per candidate
        calibration_only = bool(
            torch.as_tensor(locked_predictions[0].masks["calibration_only"]).any()
        )
        empirical = bool(torch.as_tensor(locked_predictions[0].masks["empirical_check"]).any())
        for prediction in locked_predictions:
            score = model.score(prediction, combined_raw)
            calibration_only_count += int(calibration_only)
            empirical_count += int(empirical)
            marked = _mark_check(score, calibration_only=calibration_only, empirical=empirical)
            locked_checks.append(
                _absolute_locked_check(
                    marked,
                    prediction=prediction,
                    raw=combined_raw,
                    model=model,
                    reason=(
                        "calibration-only check: a prequential verdict needs a calibration set"
                        if calibration_only
                        else (
                            "eta-fitted residual: no independent false-alarm guarantee"
                            if empirical
                            else None
                        )
                    ),
                )
            )

    # 5. encode the current RAW and score the fit-mode predictions.
    fit_context = history
    encoded_current = False
    fit_context_camera = getattr(config, "fit_context_camera", None)
    pooled: torch.Tensor | None = None
    encoded_rows: list[int] = []
    if fit_levels is not None and fit_pyramid is None:
        raise ValueError("fit_levels is only meaningful together with fit_pyramid")
    fit_levels_sliced: Any = None
    if fit_pyramid is not None:
        # One batched encode serves several decision times: the caller encoded a
        # window bundle once and hands each step its own row slice here. The
        # multi-scale evidence is sliced with the same rows, so the fit context
        # keeps the levels a per-step encode would have produced (values, valid and
        # sensor_coordinates are all sliced; level_id/support_ids are preserved).
        positions = list(range(batch)) if fit_rows is None else [int(item) for item in fit_rows]
        if len(positions) != batch:
            raise ValueError(
                f"fit_rows declares {len(positions)} rows but the bundle batch is {batch}"
            )
        source_levels = (
            fit_levels if fit_levels is not None else getattr(fit_pyramid, "levels", None)
        )
        pooled = pool_feature_pyramid(fit_pyramid, camera_id=fit_context_camera, rows=positions)
        if pooled.shape[0] != batch:
            raise ValueError(
                f"the precomputed pyramid rows yield {pooled.shape[0]} contexts but the "
                f"bundle batch is {batch}"
            )
        fit_levels_sliced = _slice_feature_levels(source_levels, positions)
        encoded_current = True
        # the slice is aligned with this bundle's local rows 0..batch-1; ``positions``
        # are the row addresses inside the caller's window pyramid
        encoded_rows = list(range(batch))
    elif getattr(model, "encoder", None) is not None and bundle.frames and new_rows:
        # Only the rows that actually form a fit prediction are encoded; a repeated
        # (``new_time_mask=False``) row never enters the forward pass and cannot
        # influence another row's features.
        encode_rows = list(new_rows)
        encode_bundle = bundle
        if len(encode_rows) < batch:
            selected = _select_bundle_rows(bundle, encode_rows)
            if selected is not None:
                encode_bundle = selected
            else:  # frames are broadcast, not batched: encode the whole bundle once
                encode_rows = list(range(batch))
        pyramid = model.encode(encode_bundle)
        pooled = pool_feature_pyramid(pyramid, camera_id=fit_context_camera)
        if pooled.shape[0] != len(encode_rows):
            raise ValueError(
                f"encoded current features have batch {pooled.shape[0]} but "
                f"{len(encode_rows)} rows were encoded"
            )
        fit_levels_sliced = getattr(pyramid, "levels", None)
        encoded_current = True
        encoded_rows = encode_rows
    if encoded_current and pooled is not None:
        fit_context = HistoryContext(
            pooled_features=pooled,
            levels=fit_levels_sliced,
            episode_ids=history.episode_ids,
            anchor_ids=history.anchor_ids,
            last_frame_index=history.last_frame_index,
            eta=history.eta,
            eta_fit_keys=history.eta_fit_keys,
        )
    fit_scores: list[Any] = []
    fit_by_row: dict[int, list[Any]] = {}
    if worlds and new_rows:
        local_of_row = {row: local for local, row in enumerate(encoded_rows)}
        for row in new_rows:
            local = local_of_row.get(row)
            row_history = (
                _row_history(fit_context, local, len(encoded_rows))
                if local is not None
                else history
            )
            fit_by_row[row] = [
                model.predict(world, row_history, row_queries[row].spec, call="fit")
                for world in worlds
            ]
        combined_raw = _concat_raw_queries([raw_queries[row] for row in new_rows])
        for index in range(len(worlds)):
            prediction = _concat_predictions([fit_by_row[row][index] for row in new_rows])
            fit_scores.append(model.score(prediction, combined_raw))

    # 6. one likelihood entry per objective; pick the best-seen world.
    duplicate_objectives = 0
    likelihood_terms_before = len(register)
    best_index: int | None = None
    best_value: float | None = None
    fit_values: list[float] = []
    if fit_scores:
        # one host transfer for the whole world vector instead of one synchronization
        # per candidate
        fit_values = [
            float(item)
            for item in torch.stack([_score_scalar_tensor(score) for score in fit_scores])
            .detach()
            .cpu()
            .tolist()
        ]
        best_value = min(fit_values)
        best_index = fit_values.index(best_value)
    objective = _objective_key(
        ledger, model, episode_ids[new_rows[-1]] if new_rows else episode_ids[0]
    )
    if fit_scores and best_value is not None:
        is_new, stored = register.add(objective, best_value)
        duplicate_objectives += 0 if is_new else 1
        best_value = stored
    scored_worlds: list[WorldHypothesis] = []
    for index, world in enumerate(worlds):
        if fit_scores:
            score = fit_scores[index]
            world.score = fit_values[index] if fit_values else _score_mean(score)
            world.score_kind = _world_score_kind(str(getattr(score, "kind", "")))
        scored_worlds.append(world)
    best_world = scored_worlds[best_index] if best_index is not None else None

    # Differentiable readout of the same fit objective: worlds[k].score is its
    # detached Python float, world_scores carries the gradient for candidate CE.
    world_scores = (
        torch.stack([_score_scalar_tensor(score) for score in fit_scores])
        if fit_scores
        else torch.empty(0)
    )

    residuals: dict[str, float] = {}
    if best_index is not None:
        residuals["fit_nll"] = (
            fit_values[best_index] if fit_values else _score_mean(fit_scores[best_index])
        )
        if locked_checks:
            residuals["prequential_nll"] = float(
                torch.stack([_score_scalar_tensor(score) for score in locked_checks])
                .detach()
                .cpu()[best_index]
            )
    cost = {
        "rows": float(batch),
        "new_time_rows": float(len(new_rows)),
        "predictions": float(sum(len(items) for items in prequential.values())),
        "locked_checks": float(len(locked_checks)),
        "fit_scores": float(len(fit_scores)),
        "calibration_only_checks": float(calibration_only_count),
        "empirical_checks": float(empirical_count),
        "new_evidence": float(new_evidence),
        "evidence_count": float(ledger.evidence_count),
        "likelihood_terms": float(len(register)),
        "likelihood_terms_before": float(likelihood_terms_before),
        "duplicate_objectives": float(duplicate_objectives),
        "updates": float(len(new_rows)),
        "targets_present": 1.0 if targets is not None else 0.0,
        "fit_context_from_current_raw": 1.0 if encoded_current else 0.0,
        "fit_context_rows": float(len(encoded_rows)),
        "fit_context_precomputed": 1.0 if fit_pyramid is not None else 0.0,
    }
    return StepOutput(
        objective=objective,
        worlds=tuple(scored_worlds),
        best_world=best_world,
        locked_checks=tuple(locked_checks),
        fit_scores=tuple(fit_scores),
        world_scores=world_scores,
        state=state,
        ledger_version=ledger.state_version,
        evidence_count=ledger.evidence_count,
        residuals=residuals,
        cost=cost,
    )

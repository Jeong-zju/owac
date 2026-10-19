"""Declared measurement density, positive variance and interference eta (P1 T2).

This module owns the measurement side of the N02 v1.1 P1 contract::

    RawPrediction(mean, variance, masks, units)
    MeasurementScore(relative_terms, raw_nll_or_energy, local_stats, kind, per_valid_scalar)
    InterferenceEstimate(log_exposure, camera_translation, noise_scale, source)

Frozen semantics (``docs/prompts/n02_v1_1/p1_interface_freeze.md`` section 3.2):

* the proper continuous term is ``0.5 * ((y - mean)**2 / variance + log(variance) + log(2*pi))``;
* a quantised observation contributes ``-log(CDF(upper) - CDF(lower))``, evaluated in the
  log domain so a deep-tail difference never underflows to ``log(0)``;
* a clipped (saturated) sample contributes the negative log of the corresponding tail
  probability, and that tail probability is recorded;
* ``valid = ~(missing | bad)`` and missing/bad samples contribute no term at all;
* variance is positive-parameterised; the analytic sensor variance and the model
  predictive uncertainty are reported separately and only summed for the score;
* a synthetic model with a declared density is scored as a proper NLL; an unverified real
  pseudo-RAW model is scored as ``empirical_energy`` and never claims a density;
* every real sample position contributes at most once: duplicated sample identities are
  dropped before scoring and counted;
* interference ``eta`` is estimated from history-declared records only.  The history
  estimator refuses evaluator-side objects and always returns ``source="history"``; an
  oracle estimate can only be constructed explicitly on the evaluator side.

**Unit semantics (frozen P1 section 3.8).**  Everything this module computes - the
Gaussian/quantised density, the clip tail probabilities, the variance guards, the
quantisation step and the sensor shot/read declaration - is evaluated in the query's
declared ``units`` (variance in ``units**2``).  This module performs **no** internal
normalisation and owns **no** normalised-signal affine: the black/white/scale affine and
its ``calibration_version`` are declared by the factors side (T3 ``UnitAffine``), which
converts its internal domain into the declared units before building a prediction.
:meth:`SensorModel.unit_semantics` exposes the full parameter/unit table for the manifest.
A prediction whose numbers live in another domain must therefore be either rejected by the
unit string or produce a clearly large negative score, never a silently small one.

No function in this module reads an ``EvaluationTruth``.  Camera motion translates every
projected point (background and objects); entity motion translates only that entity's own
projected points.  ``factors.py`` (T3) consumes this module through :class:`SensorModel`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np
import torch

from .synthetic import quantisation_interval_probability, saturation_tail_probability

__all__ = [
    "INTERFERENCE_SOURCES",
    "MASK_NAMES",
    "MAX_VARIANCE",
    "MIN_VARIANCE",
    "SCORE_KINDS",
    "DEFAULT_INTERFERENCE_PRIOR",
    "DensityVerificationError",
    "InterferenceEstimate",
    "InterferenceObservation",
    "InterferencePrior",
    "MeasurementScore",
    "RawPrediction",
    "SensorModel",
    "UnitsMismatchError",
    "clip_tail_log_prob",
    "declared_quantisation_interval_probability",
    "declared_saturation_tail_probability",
    "deduplicate_samples",
    "estimate_interference",
    "exp_variance",
    "gaussian_nll",
    "interval_log_probability",
    "positive_variance",
    "predictive_variance",
    "project_camera_motion",
    "project_entity_motion",
    "quantisation_log_prob",
    "robust_energy",
    "sanitise_variance",
    "score_diagnostics",
    "score_prediction",
    "sensor_variance",
    "softplus_variance",
    "standardised_residual",
]

#: Score kinds a :class:`MeasurementScore` may declare.  ``proper_nll`` is a declared
#: normalised density; ``empirical_energy`` is an unnormalised energy and never supports
#: posterior claims.
SCORE_KINDS = ("proper_nll", "empirical_energy")

#: Query masks carried by :class:`RawPrediction`/scoring.
MASK_NAMES = ("valid", "bad", "missing", "saturated")

#: Interference estimate provenance; only the evaluator may construct ``oracle``.
INTERFERENCE_SOURCES = ("history", "oracle")

#: Numerical guard rails for the variance.  They bound floating point failure modes
#: (a zero or infinite variance), they are not a model prior.
MIN_VARIANCE = 1e-12
MAX_VARIANCE = 1e30

_LOG_2PI = math.log(2.0 * math.pi)
_HALF_LOG_2PI = 0.5 * _LOG_2PI
_LOG_2 = math.log(2.0)

#: Where the central ``log_ndtr`` meets the Mills-ratio asymptotic tails.
_TAIL_CLAMP = 12.0

#: Tokens that mark an object as evaluator-side truth; the history estimator refuses any
#: input carrying them so eta can never be read from ``EvaluationTruth``.
_FORBIDDEN_TOKENS = ("truth", "oracle", "evaluation", "label", "future", "target")

_HUBER_KAPPA = 3.0


class UnitsMismatchError(ValueError):
    """Raised when a prediction and a query do not declare the identical unit string."""


class DensityVerificationError(ValueError):
    """Raised when a proper normalised density is requested but not declared verified."""


# --------------------------------------------------------------------------------------
# positive variance and analytic sensor variance
# --------------------------------------------------------------------------------------


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, np.generic):
        tensor = torch.as_tensor(value.item())
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def sanitise_variance(
    variance: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Clamp a variance into ``[min_variance, max_variance]`` as a finite positive value.

    Non-finite entries are mapped to the corresponding bound (``nan``/``+inf`` to the
    upper bound, ``-inf`` to the lower bound) so a single bad variance can never produce
    ``NaN``/``Inf`` in a score.
    """
    if not min_variance > 0.0:
        raise ValueError("min_variance must be positive")
    if not max_variance >= min_variance:
        raise ValueError("max_variance must be >= min_variance")
    tensor = _as_tensor(variance)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    safe = torch.nan_to_num(
        tensor, nan=float(max_variance), posinf=float(max_variance), neginf=float(min_variance)
    )
    return safe.clamp(min=float(min_variance), max=float(max_variance))


def softplus_variance(
    raw: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Positive variance from an unconstrained parameter: ``softplus(raw)``."""
    raw_tensor = _as_tensor(raw)
    if not raw_tensor.is_floating_point():
        raw_tensor = raw_tensor.to(torch.get_default_dtype())
    return sanitise_variance(
        torch.nn.functional.softplus(raw_tensor),
        min_variance=min_variance,
        max_variance=max_variance,
    )


def exp_variance(
    raw: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Positive variance from an unconstrained parameter: ``exp(raw)``."""
    raw_tensor = _as_tensor(raw)
    if not raw_tensor.is_floating_point():
        raw_tensor = raw_tensor.to(torch.get_default_dtype())
    return sanitise_variance(
        torch.exp(raw_tensor), min_variance=min_variance, max_variance=max_variance
    )


def positive_variance(
    raw: Any,
    *,
    parameterisation: str = "softplus",
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Positive variance with the declared parameterisation (``softplus`` or ``exp``)."""
    if parameterisation == "softplus":
        return softplus_variance(raw, min_variance=min_variance, max_variance=max_variance)
    if parameterisation == "exp":
        return exp_variance(raw, min_variance=min_variance, max_variance=max_variance)
    raise ValueError(f"unknown variance parameterisation: {parameterisation!r}")


def sensor_variance(
    mean: Any,
    *,
    shot_noise_scale: float,
    read_noise_signal: float = 0.0,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """Declared Poisson-Gaussian sensor variance in signal units.

    ``shot_noise_scale`` is electrons per unit of normalised signal, so the shot variance
    is ``max(mean, 0) / shot_noise_scale``; ``read_noise_signal`` is a read-noise standard
    deviation in signal units.  ``noise_scale`` is the interference standard-deviation
    multiplier, hence the variance scales with its square.
    """
    if not shot_noise_scale > 0.0:
        raise ValueError("shot_noise_scale must be positive")
    if read_noise_signal < 0.0:
        raise ValueError("read_noise_signal must be non-negative")
    if not noise_scale > 0.0:
        raise ValueError("noise_scale must be positive")
    mean_tensor = _as_tensor(mean)
    if not mean_tensor.is_floating_point():
        mean_tensor = mean_tensor.to(torch.get_default_dtype())
    shot = mean_tensor.clamp(min=0.0) / float(shot_noise_scale)
    read = float(read_noise_signal) ** 2
    return (float(noise_scale) ** 2) * (shot + read)


def predictive_variance(
    sensor_var: Any,
    model_var: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Total predictive variance: sensor variance plus model predictive uncertainty."""
    sensor_tensor = _as_tensor(sensor_var).clamp(min=0.0)
    model_tensor = _as_tensor(model_var).clamp(min=0.0)
    return sanitise_variance(
        sensor_tensor + model_tensor, min_variance=min_variance, max_variance=max_variance
    )


# --------------------------------------------------------------------------------------
# likelihood terms
# --------------------------------------------------------------------------------------


def standardised_residual(
    values: Any,
    mean: Any,
    variance: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """``(y - mean) / sqrt(variance)`` with the finite positive variance guard."""
    var = sanitise_variance(variance, min_variance=min_variance, max_variance=max_variance)
    return (_as_tensor(values) - _as_tensor(mean)) / torch.sqrt(var)


def gaussian_nll(
    values: Any,
    mean: Any,
    variance: Any,
    *,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """Proper Gaussian negative log density ``0.5*((y-mu)^2/v + log v + log 2pi)``.

    The ``log(v)`` term is what stops "variance goes to infinity" from being a free way to
    improve the score: past ``v = (y-mu)^2`` the term grows monotonically.
    """
    var = sanitise_variance(variance, min_variance=min_variance, max_variance=max_variance)
    delta = _as_tensor(values) - _as_tensor(mean)
    return 0.5 * (delta * delta / var + torch.log(var) + _LOG_2PI)


def _log_ndtr(values: torch.Tensor) -> torch.Tensor:
    """``log(CDF(values))`` with finite analytic gradients for every finite input.

    ``torch.special.log_ndtr`` is accurate in the central region, but its backward is
    numerically invalid for very large ``|x|`` in float32 (it returns 0, ``inf`` or a
    constant depending on the exact argument), which would turn the gradient of a far-off
    prediction into ``NaN``.  The central branch is therefore fed a clamped argument and
    the far tails use the Mills-ratio asymptotic with its first three corrections, so no
    branch can contribute a non-finite value or gradient.
    """
    x = torch.as_tensor(values)
    if not x.is_floating_point():
        x = x.to(torch.get_default_dtype())
    central = torch.special.log_ndtr(x.clamp(-_TAIL_CLAMP, _TAIL_CLAMP))
    lower_z = x.clamp_max(-_TAIL_CLAMP)
    upper_z = x.clamp_min(_TAIL_CLAMP)
    inverse_square = 1.0 / (lower_z * lower_z)
    mills_lower = torch.log(
        1.0 - inverse_square * (1.0 - inverse_square * (3.0 - 15.0 * inverse_square))
    )
    lower_tail = -0.5 * x * x - torch.log(-lower_z) - _HALF_LOG_2PI + mills_lower
    inverse_square_upper = 1.0 / (upper_z * upper_z)
    mills_upper = torch.log(
        1.0
        - inverse_square_upper * (1.0 - inverse_square_upper * (3.0 - 15.0 * inverse_square_upper))
    )
    log_survival = -0.5 * x * x - torch.log(upper_z) - _HALF_LOG_2PI + mills_upper
    upper_tail = torch.log1p(-torch.exp(log_survival))
    return torch.where(
        x < -_TAIL_CLAMP, lower_tail, torch.where(x > _TAIL_CLAMP, upper_tail, central)
    )


def interval_log_probability(lower: Any, upper: Any) -> torch.Tensor:
    """``log(CDF(upper) - CDF(lower))``, stable for broad, narrow and deep-tail intervals.

    Three regimes are combined, each used where it is accurate and kept finite so its
    gradient never contaminates the selected regime:

    * the log-CDF difference ``log CDF(upper) + log1mexp(delta)`` (exact, and used
      whenever ``delta`` is above its floating point cancellation floor);
    * the log-survival difference ``log sf(lower) + log1mexp(log sf(lower) -
      log sf(upper))`` when the log-CDF difference has cancelled but the survival
      difference is resolved (a deep tail where both endpoints round to ``CDF == 1``);
    * the narrow-interval expansion ``log(width) + log phi(centre)`` as the finite
      fallback (relative error ``O(width**2 (1 + centre**2))``).
    """
    lower_tensor = _as_tensor(lower)
    upper_tensor = _as_tensor(upper)
    dtype = torch.promote_types(lower_tensor.dtype, upper_tensor.dtype)
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()
    lower_tensor = lower_tensor.to(dtype)
    upper_tensor = upper_tensor.to(dtype)

    eps = float(torch.finfo(dtype).eps)
    tiny = torch.finfo(dtype).tiny
    log_lower = _log_ndtr(lower_tensor)
    log_upper = _log_ndtr(upper_tensor)
    delta = (log_upper - log_lower).clamp_min(0.0)

    centre = 0.5 * (upper_tensor + lower_tensor)
    width = (upper_tensor - lower_tensor).clamp_min(tiny)
    narrow = torch.log(width) - 0.5 * centre * centre - _HALF_LOG_2PI

    log_survival_lower = _log_ndtr(-lower_tensor)
    log_survival_upper = _log_ndtr(-upper_tensor)
    survival_delta = (log_survival_lower - log_survival_upper).clamp_min(0.0)

    # A branch is selected only while its own difference is resolved by the floating
    # point resolution; the narrow-interval expansion is the finite fallback.
    resolution = 16.0 * eps
    general_floor = resolution * torch.maximum(
        torch.ones_like(delta), torch.maximum(log_lower.abs(), log_upper.abs())
    )
    tail_floor = resolution * torch.maximum(
        torch.ones_like(survival_delta),
        torch.maximum(log_survival_lower.abs(), log_survival_upper.abs()),
    )
    general_valid = delta >= general_floor
    tail_valid = survival_delta >= tail_floor
    use_general = general_valid
    use_tail = ~general_valid & tail_valid

    delta_for_general = torch.where(use_general, delta, torch.ones_like(delta))
    log_one_minus_exp = torch.where(
        delta_for_general > _LOG_2,
        torch.log1p(-torch.exp(-delta_for_general)),
        torch.log(-torch.expm1(-delta_for_general)),
    )
    general = log_upper + log_one_minus_exp

    survival_delta_for_tail = torch.where(use_tail, survival_delta, torch.ones_like(survival_delta))
    tail_factor = torch.where(
        survival_delta_for_tail > _LOG_2,
        torch.log1p(-torch.exp(-survival_delta_for_tail)),
        torch.log(-torch.expm1(-survival_delta_for_tail)),
    )
    tail = log_survival_lower + tail_factor

    return torch.where(use_general, general, torch.where(use_tail, tail, narrow))


def quantisation_log_prob(
    values: Any,
    mean: Any,
    variance: Any,
    *,
    step: float,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """``log P(value - step/2 <= noisy sample < value + step/2)`` in the log domain."""
    if not step > 0.0:
        raise ValueError("step must be positive")
    var = sanitise_variance(variance, min_variance=min_variance, max_variance=max_variance)
    sigma = torch.sqrt(var)
    centre = _as_tensor(values) - _as_tensor(mean)
    half = 0.5 * float(step)
    return interval_log_probability((centre - half) / sigma, (centre + half) / sigma)


def clip_tail_log_prob(
    mean: Any,
    variance: Any,
    *,
    level: float,
    side: str,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
) -> torch.Tensor:
    """``log P(noisy sample >= level)`` (``side="top"``) or ``<= level`` (``"bottom"``).

    A clipped sample is a censored observation: its contribution is the tail probability
    at the clip level, never the clipped value treated as exact.
    """
    if side not in ("top", "bottom"):
        raise ValueError(f"side must be 'top' or 'bottom', got {side!r}")
    var = sanitise_variance(variance, min_variance=min_variance, max_variance=max_variance)
    sigma = torch.sqrt(var)
    offset = (_as_tensor(mean) - float(level)) / sigma
    if side == "top":
        return _log_ndtr(offset)
    return _log_ndtr(-offset)


def robust_energy(
    standardised: Any,
    *,
    kappa: float = _HUBER_KAPPA,
) -> torch.Tensor:
    """Huber robust energy of a standardised residual (empirical, unnormalised)."""
    if not kappa > 0.0:
        raise ValueError("kappa must be positive")
    abs_z = _as_tensor(standardised).abs()
    return torch.where(
        abs_z <= kappa,
        0.5 * abs_z * abs_z,
        kappa * abs_z - 0.5 * kappa * kappa,
    )


def _huber_energy(standardised: torch.Tensor) -> torch.Tensor:
    return robust_energy(standardised)


# --------------------------------------------------------------------------------------
# sample identity: each real sampling position contributes at most once
# --------------------------------------------------------------------------------------


def _identity_key(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.detach().flatten().tolist())
    if isinstance(value, np.ndarray):
        return tuple(value.flatten().tolist())
    return value


def deduplicate_samples(
    sample_ids: Sequence[Any] | None = None,
    coordinates: Any | None = None,
    *,
    decimals: int = 4,
) -> tuple[torch.Tensor, int]:
    """First-occurrence indices of unique sample identities plus the dropped count.

    ``sample_ids`` is the persistent identity (a retransmission repeats it); ``coordinates``
    is a fallback identity for overlapping multi-scale samples.  Exactly one of the two is
    used, and the coordinate key is rounded at ``decimals`` so a sample that appears in two
    scales at the same sensor position is one position.
    """
    if (sample_ids is None) == (coordinates is None):
        raise ValueError("pass exactly one of sample_ids or coordinates")
    if sample_ids is not None:
        keys = [_identity_key(item) for item in sample_ids]
    else:
        array = np.asarray(coordinates, dtype=np.float64)
        if array.ndim < 2 or array.shape[-1] < 2:
            raise ValueError("coordinates must have shape (..., >=2)")
        flat = array.reshape(-1, array.shape[-1])[:, :2]
        scaled = np.rint(flat * float(10**decimals)).astype(np.int64)
        keys = [tuple(int(item) for item in row) for row in scaled]
    seen: set = set()
    indices: list[int] = []
    dropped = 0
    for position, key in enumerate(keys):
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        indices.append(position)
    return torch.tensor(indices, dtype=torch.long), dropped


# --------------------------------------------------------------------------------------
# prediction and score records
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class RawPrediction:
    """One candidate's declared measurement prediction at the query sample points.

    ``variance`` is the total positive predictive variance.  ``variance_components`` is an
    optional split (``sensor``/``model``/``total``) required by the contract's "sensor
    variance and model predictive uncertainty are listed separately"; it never replaces the
    frozen ``variance`` field.  ``masks`` carries ``valid``/``bad``/``missing``/``saturated``
    when the prediction knows the query's sampling condition.
    """

    mean: torch.Tensor
    variance: torch.Tensor
    masks: dict[str, torch.Tensor]
    units: str
    variance_components: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = _as_tensor(self.mean)
        self.variance = _as_tensor(self.variance)
        if self.mean.shape != self.variance.shape:
            raise ValueError(
                f"mean shape {tuple(self.mean.shape)} != variance shape "
                f"{tuple(self.variance.shape)}"
            )
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a non-empty string")
        if not isinstance(self.masks, dict):
            raise TypeError("masks must be a dict of name -> tensor")
        self.masks = {str(name): _as_tensor(mask) for name, mask in self.masks.items()}
        self.variance_components = {
            str(name): _as_tensor(value) for name, value in dict(self.variance_components).items()
        }


@dataclass(slots=True)
class MeasurementScore:
    """One prediction's declared score against one query measurement.

    ``raw_nll_or_energy`` is the sum over valid scalars; ``per_valid_scalar`` is that sum
    divided by the number of valid scalars (the training loss view).  ``kind`` must be a
    real score type: a proper NLL is a normalised density, an empirical energy is not, and
    the two must not be mixed silently.
    """

    relative_terms: dict[str, torch.Tensor]
    raw_nll_or_energy: torch.Tensor
    local_stats: dict[str, torch.Tensor]
    kind: str
    per_valid_scalar: torch.Tensor

    def __post_init__(self) -> None:
        if self.kind not in SCORE_KINDS:
            raise ValueError(f"kind must be one of {SCORE_KINDS}, got {self.kind!r}")
        self.raw_nll_or_energy = _as_tensor(self.raw_nll_or_energy).reshape(())
        self.per_valid_scalar = _as_tensor(self.per_valid_scalar).reshape(())
        self.relative_terms = {
            str(name): _as_tensor(value) for name, value in dict(self.relative_terms).items()
        }
        self.local_stats = {
            str(name): _as_tensor(value) for name, value in dict(self.local_stats).items()
        }


@dataclass(slots=True)
class InterferenceEstimate:
    """Declared imaging interference eta for one decision moment.

    ``log_exposure`` is shared across all cameras, ``camera_translation`` is a
    ``(frame, camera, 2)`` tensor in sensor pixels, and ``noise_scale`` multiplies the
    standard deviation of the declared sensor variance.  ``source`` is ``"history"`` for
    everything :func:`estimate_interference` produces; only evaluator-side code may
    construct ``source="oracle"`` explicitly from the reference interference.
    """

    log_exposure: torch.Tensor
    camera_translation: torch.Tensor
    noise_scale: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        self.log_exposure = _as_tensor(self.log_exposure, dtype=torch.float64).reshape(())
        self.camera_translation = _as_tensor(self.camera_translation, dtype=torch.float64)
        self.noise_scale = _as_tensor(self.noise_scale, dtype=torch.float64).reshape(())
        if self.camera_translation.ndim != 3 or self.camera_translation.shape[-1] != 2:
            raise ValueError(
                "camera_translation must have shape (frames, cameras, 2), got "
                f"{tuple(self.camera_translation.shape)}"
            )
        if self.source not in INTERFERENCE_SOURCES:
            raise ValueError(f"source must be one of {INTERFERENCE_SOURCES}, got {self.source!r}")


# --------------------------------------------------------------------------------------
# query coercion and scoring
# --------------------------------------------------------------------------------------


def _coerce_query(values: Any) -> tuple[Any, str | None, dict[str, Any]]:
    """Accept a tensor/array or a declared query object with values, unit and masks."""
    if isinstance(values, Mapping) and "values" in values:
        payload = dict(values)
        raw = payload.pop("values")
        units = payload.pop("unit", payload.pop("units", None))
        masks: dict[str, Any] = {}
        for name in MASK_NAMES:
            if name in payload:
                masks[name] = payload.pop(name)
        extra_masks = payload.pop("masks", None)
        if isinstance(extra_masks, Mapping):
            masks.update(extra_masks)
        return raw, None if units is None else str(units), masks
    if not isinstance(values, (torch.Tensor, np.ndarray, Mapping)) and hasattr(values, "values"):
        raw = values.values
        units = getattr(values, "units", None)
        if units is None:
            units = getattr(values, "unit", None)
        masks = {}
        for name in MASK_NAMES:
            attribute = getattr(values, f"{name}_mask", None)
            if attribute is not None:
                masks[name] = attribute
        extra_masks = getattr(values, "masks", None)
        if isinstance(extra_masks, Mapping):
            masks.update(extra_masks)
        return raw, None if units is None else str(units), masks
    return values, None, {}


def _broadcast_mask(mask: Any, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    tensor = _as_tensor(mask, dtype=torch.bool)
    return tensor.to(device=device).broadcast_to(shape)


def score_prediction(
    prediction: RawPrediction,
    values: Any,
    *,
    masks: Mapping[str, Any] | None = None,
    units: str | None = None,
    kind: str = "proper_nll",
    sample_ids: Sequence[Any] | None = None,
    coordinates: Any | None = None,
    quantisation_step: float | None = None,
    saturation_level: float | None = None,
    floor_level: float | None = None,
    clip_top: Any | None = None,
    clip_bottom: Any | None = None,
    min_variance: float = MIN_VARIANCE,
    max_variance: float = MAX_VARIANCE,
    deduplicate: bool = True,
) -> MeasurementScore:
    """Score one :class:`RawPrediction` against one query measurement.

    ``values`` may be a tensor/array or a declared query object carrying ``values`` and
    ``unit``/``units`` plus ``*_mask`` attributes (for example ``SensorMeasurement``).
    ``units`` must equal ``prediction.units`` verbatim, otherwise :class:`UnitsMismatchError`
    is raised and no comparison happens.

    ``quantisation_step`` selects the quantised observation model; ``None`` uses the
    continuous Gaussian density.  Clipped samples must declare ``saturation_level``
    (top) / ``floor_level`` (bottom) or an explicit ``clip_top``/``clip_bottom`` mask, and
    contribute a tail probability.  ``kind="empirical_energy"`` scores an unnormalised
    robust energy instead of a density.
    """
    if not isinstance(prediction, RawPrediction):
        raise TypeError(f"prediction must be a RawPrediction, got {type(prediction).__name__}")
    if kind not in SCORE_KINDS:
        raise ValueError(f"kind must be one of {SCORE_KINDS}, got {kind!r}")

    raw_values, derived_units, derived_masks = _coerce_query(values)
    query_units = units if units is not None else derived_units
    if query_units is None:
        raise ValueError("query units are required: pass units=... or a query carrying unit(s)")
    if str(query_units) != str(prediction.units):
        raise UnitsMismatchError(
            f"prediction units {prediction.units!r} != query units {str(query_units)!r}"
        )

    mean_tensor = _as_tensor(prediction.mean)
    variance_tensor = _as_tensor(prediction.variance)
    values_tensor = _as_tensor(raw_values)
    compute_dtype = torch.promote_types(
        torch.promote_types(mean_tensor.dtype, variance_tensor.dtype), values_tensor.dtype
    )
    if not compute_dtype.is_floating_point:
        compute_dtype = torch.get_default_dtype()
    mean_tensor = mean_tensor.to(compute_dtype)
    variance_tensor = variance_tensor.to(compute_dtype)
    values_tensor = values_tensor.to(compute_dtype)
    mean_tensor, variance_tensor, values_tensor = torch.broadcast_tensors(
        mean_tensor, variance_tensor, values_tensor
    )
    shape = tuple(mean_tensor.shape)
    device = mean_tensor.device

    variance_floored = (
        (variance_tensor < float(min_variance))
        | (variance_tensor > float(max_variance))
        | ~torch.isfinite(variance_tensor)
    )
    variance = sanitise_variance(
        variance_tensor, min_variance=min_variance, max_variance=max_variance
    )
    sigma = torch.sqrt(variance)
    finite_values = torch.isfinite(values_tensor)
    standardised = (values_tensor - mean_tensor) / sigma

    sources: list[Mapping[str, Any]] = [prediction.masks]
    if derived_masks:
        sources.append(derived_masks)
    if masks:
        sources.append(masks)
    merged: dict[str, torch.Tensor] = {}
    for name in ("bad", "missing", "saturated"):
        merged[name] = torch.zeros(shape, dtype=torch.bool, device=device)
        for source in sources:
            if source and source.get(name) is not None:
                merged[name] = merged[name] | _broadcast_mask(source[name], shape, device)
    valid = torch.ones(shape, dtype=torch.bool, device=device)
    for source in sources:
        if source and source.get("valid") is not None:
            valid = valid & _broadcast_mask(source["valid"], shape, device)
    valid = valid & ~merged["missing"] & ~merged["bad"] & finite_values

    saturated = merged["saturated"]
    top = torch.zeros(shape, dtype=torch.bool, device=device)
    bottom = torch.zeros(shape, dtype=torch.bool, device=device)
    if clip_top is not None:
        top = top | _broadcast_mask(clip_top, shape, device)
    if clip_bottom is not None:
        bottom = bottom | _broadcast_mask(clip_bottom, shape, device)
    if clip_top is None and clip_bottom is None and bool(saturated.any()):
        if saturation_level is not None and floor_level is not None:
            top = saturated & (values_tensor >= float(saturation_level))
            bottom = saturated & (values_tensor <= float(floor_level))
            if bool((saturated & ~top & ~bottom).any()):
                raise ValueError("saturated samples lie outside the declared clip levels")
        elif saturation_level is not None:
            top = saturated
        elif floor_level is not None:
            bottom = saturated
        else:
            raise ValueError(
                "saturated samples are present but no clip level or clip mask was declared"
            )
    if bool((top & bottom).any()):
        raise ValueError("a sample cannot be clipped at both the top and the bottom")
    if bool(top.any()) and saturation_level is None:
        raise ValueError("top-clipped samples require saturation_level")
    if bool(bottom.any()) and floor_level is None:
        raise ValueError("bottom-clipped samples require floor_level")

    counted = valid & ~top & ~bottom
    scored = counted | (top & valid) | (bottom & valid)

    duplicate_count = 0
    total_scalars = int(values_tensor.numel())
    if deduplicate and (sample_ids is not None or coordinates is not None):
        if sample_ids is not None and len(sample_ids) != total_scalars:
            raise ValueError(
                f"sample_ids has {len(sample_ids)} entries for {total_scalars} scalars"
            )
        indices, duplicate_count = deduplicate_samples(
            sample_ids=sample_ids, coordinates=coordinates
        )
        keep = torch.zeros(total_scalars, dtype=torch.bool, device=device)
        keep[indices.to(device)] = True
        keep = keep.reshape(shape)
        counted = counted & keep
        scored = scored & keep
        valid = valid & keep

    zero = torch.zeros_like(standardised)
    if kind == "proper_nll":
        if quantisation_step is None:
            base = 0.5 * (standardised * standardised + torch.log(variance) + _LOG_2PI)
            base_name = "continuous"
        else:
            half_over_sigma = (0.5 * float(quantisation_step)) / sigma
            base = -interval_log_probability(
                standardised - half_over_sigma, standardised + half_over_sigma
            )
            base_name = "quantisation"
        term = torch.where(counted, base, zero)
        relative_terms = {base_name: torch.where(counted, base, zero)}
        local_stats_extra: dict[str, Any] = {}
        if bool(top.any()):
            top_log_probability = clip_tail_log_prob(
                mean_tensor,
                variance,
                level=float(saturation_level),
                side="top",
                min_variance=min_variance,
                max_variance=max_variance,
            )
            top_nll = -top_log_probability
            term = torch.where(top & valid, top_nll, term)
            relative_terms["clip_top"] = torch.where(top & valid, top_nll, zero)
            local_stats_extra["top_clip_tail_probability_max"] = torch.where(
                top & valid, torch.exp(top_log_probability), zero
            ).max()
        if bool(bottom.any()):
            bottom_log_probability = clip_tail_log_prob(
                mean_tensor,
                variance,
                level=float(floor_level),
                side="bottom",
                min_variance=min_variance,
                max_variance=max_variance,
            )
            bottom_nll = -bottom_log_probability
            term = torch.where(bottom & valid, bottom_nll, term)
            relative_terms["clip_bottom"] = torch.where(bottom & valid, bottom_nll, zero)
            local_stats_extra["bottom_clip_tail_probability_max"] = torch.where(
                bottom & valid, torch.exp(bottom_log_probability), zero
            ).max()
    else:
        energy = _huber_energy(standardised)
        term = torch.where(scored, energy, zero)
        relative_terms = {
            "energy": term,
            "standardised_residual": torch.where(scored, standardised, zero),
        }
        local_stats_extra = {}

    total = term.sum()
    n_valid = scored.sum()
    n_valid_float = n_valid.to(compute_dtype)
    denominator = n_valid_float.clamp_min(1.0)
    abs_standardised = torch.where(scored, standardised.abs(), zero)
    empty_stats = (
        {
            "abs_standardised_residual_max": torch.zeros((), dtype=compute_dtype, device=device),
            "min_variance_used": torch.zeros((), dtype=compute_dtype, device=device),
            "max_variance_used": torch.zeros((), dtype=compute_dtype, device=device),
        }
        if total_scalars == 0
        else {
            "abs_standardised_residual_max": abs_standardised.max(),
            "min_variance_used": variance.min(),
            "max_variance_used": variance.max(),
        }
    )

    local_stats: dict[str, Any] = {
        "standardised_residual": torch.where(scored, standardised, zero),
        "valid": scored,
        "abs_standardised_residual_mean": abs_standardised.sum() / denominator,
        "abs_standardised_residual_max": empty_stats["abs_standardised_residual_max"],
        "n_valid_scalars": n_valid_float,
        "n_invalid_scalars": torch.as_tensor(total_scalars, dtype=compute_dtype, device=device)
        - n_valid_float,
        "n_nonfinite_values": (~finite_values).sum().to(compute_dtype),
        "n_clipped_top": (top & scored).sum().to(compute_dtype),
        "n_clipped_bottom": (bottom & scored).sum().to(compute_dtype),
        "n_duplicate_samples": torch.as_tensor(duplicate_count, dtype=compute_dtype, device=device),
        "n_variance_floored": variance_floored.sum().to(compute_dtype),
        "min_variance_used": empty_stats["min_variance_used"],
        "max_variance_used": empty_stats["max_variance_used"],
    }
    local_stats.update(local_stats_extra)

    return MeasurementScore(
        relative_terms=relative_terms,
        raw_nll_or_energy=total,
        local_stats=local_stats,
        kind=kind,
        per_valid_scalar=total / denominator,
    )


# --------------------------------------------------------------------------------------
# score diagnostics for the run manifest
# --------------------------------------------------------------------------------------


def score_diagnostics(
    score: MeasurementScore,
    prediction: RawPrediction,
    *,
    scale_suspect_threshold: float = 1e3,
) -> dict[str, Any]:
    """Magnitude and scale summary of one score, for the run manifest.

    Reports the NLL magnitude, variance quantiles and standardised-residual quantiles over
    the scored samples.  ``scale_suspect`` is True when the *median* absolute standardised
    residual exceeds ``scale_suspect_threshold`` (1000 by default): a sigma that large is
    a unit/scale mistake rather than a wrong candidate, and this is the cheap detector for
    "a normalised prediction scored against DN values".
    """
    if not isinstance(score, MeasurementScore) or not isinstance(prediction, RawPrediction):
        raise TypeError("score_diagnostics expects a MeasurementScore and a RawPrediction")
    valid = torch.as_tensor(score.local_stats.get("valid", torch.ones(1, dtype=torch.bool)))
    residual = torch.as_tensor(score.local_stats.get("standardised_residual", torch.zeros(1)))
    abs_residual = residual.abs().reshape(-1).to(torch.float64)
    flat_valid = valid.reshape(-1).bool()
    if flat_valid.numel() == abs_residual.numel():
        abs_residual = abs_residual[flat_valid]
    variance = torch.as_tensor(prediction.variance, dtype=torch.float64).reshape(-1)

    def _quantile(values: torch.Tensor, level: float) -> float:
        if values.numel() == 0:
            return float("nan")
        return float(torch.quantile(values.detach(), level))

    median_residual = _quantile(abs_residual, 0.5)
    return {
        "kind": score.kind,
        "raw_nll_or_energy": float(score.raw_nll_or_energy.detach()),
        "per_valid_scalar": float(score.per_valid_scalar.detach()),
        "n_valid_scalars": float(
            torch.as_tensor(score.local_stats.get("n_valid_scalars", 0.0)).detach()
        ),
        "abs_standardised_residual_median": median_residual,
        "abs_standardised_residual_p95": _quantile(abs_residual, 0.95),
        "variance_median": _quantile(variance, 0.5),
        "variance_p95": _quantile(variance, 0.95),
        "scale_suspect": bool(
            math.isfinite(median_residual) and median_residual > float(scale_suspect_threshold)
        ),
    }


# --------------------------------------------------------------------------------------
# declared sensor model (T2 -> T3 boundary)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class SensorModel:
    """The declared measurement density T3 passes into ``ObservationModel``.

    **Unit semantics (frozen P1 section 3.8).**  Every density statement is evaluated in
    the query's declared ``units``; nothing is silently normalised.

    ==========================  =========================================================
    parameter                   unit
    ==========================  =========================================================
    ``units``                   the one unit string every prediction and query must match
    ``quantisation_step``       ``units``; the interval is ``value +/- step/2``
    ``saturation_level``        ``units``; the top clip point (for example white 4095 DN)
    ``floor_level``             ``units``; the bottom clip point (for example black 64 DN)
    ``min_variance``            ``units**2``; numerical floor of the predictive variance
    ``max_variance``            ``units**2``; numerical ceiling of the predictive variance
    ``noise_scale``             dimensionless standard-deviation multiplier (interference)
    ``shot_noise_scale``        electrons per unit of the declared signal value
    ``read_noise_signal``       read-noise sigma in ``units``
    ==========================  =========================================================

    **No internal normalisation.**  Every quantity above is interpreted in the declared
    ``units`` exactly as passed; this module never maps a normalised-domain value or
    variance onto a declared-unit query.  The normalised-signal affine (black level, white
    level, scale) is declared on the *factors* side (T3 ``UnitAffine``): T3 converts its
    internal domain to the declared units before calling :meth:`build_prediction`, and
    :meth:`sensor_variance` then evaluates the declared shot/read statement at that
    declared mean.  A DN model therefore never applies a normalised-domain variance to a
    DN value.  For a sensor whose declared unit carries a black offset, the mean passed to
    :meth:`sensor_variance` is the value the shot declaration is written against (the
    signal above black), matching how the declaration was calibrated.

    ``shot_noise_scale``/``read_noise_signal`` may be declared to mirror
    ``synthetic.MeasurementModel`` once T3 has converted them into the declared units
    (``shot_noise_scale_units = shot_noise_scale_normalised / scale``,
    ``read_noise_signal_units = read_noise_signal_normalised * scale``); the affine and
    its ``calibration_version`` belong to the factors-side declaration.

    ``density_verified`` is ``True`` only for a synthetic model whose joint density is
    declared; real pseudo-RAW with an unverifiable noise structure must set it ``False``
    and is then scored as ``empirical_energy``.
    """

    units: str
    quantisation_step: float | None = None
    saturation_level: float | None = None
    floor_level: float | None = None
    shot_noise_scale: float | None = None
    read_noise_signal: float = 0.0
    noise_scale: float = 1.0
    density_verified: bool = True
    min_variance: float = MIN_VARIANCE
    max_variance: float = MAX_VARIANCE

    def __post_init__(self) -> None:
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("units must be a non-empty string")
        if self.quantisation_step is not None and not self.quantisation_step > 0.0:
            raise ValueError("quantisation_step must be positive or None")
        if self.shot_noise_scale is not None and not self.shot_noise_scale > 0.0:
            raise ValueError("shot_noise_scale must be positive or None")
        if self.read_noise_signal < 0.0:
            raise ValueError("read_noise_signal must be non-negative")
        if not self.noise_scale > 0.0:
            raise ValueError("noise_scale must be positive")
        sanitise_variance(0.0, min_variance=self.min_variance, max_variance=self.max_variance)

    @property
    def kind(self) -> str:
        """Declared score kind; an unverified density can only be an empirical energy."""
        return "proper_nll" if self.density_verified else "empirical_energy"

    def unit_semantics(self) -> dict[str, Any]:
        """JSON-able unit declaration of every parameter, for the run manifest."""
        units = str(self.units)
        return {
            "units": units,
            "variance_unit": f"{units}^2",
            "quantisation_step": self.quantisation_step,
            "quantisation_step_unit": units,
            "saturation_level": self.saturation_level,
            "saturation_level_unit": units,
            "floor_level": self.floor_level,
            "floor_level_unit": units,
            "min_variance": float(self.min_variance),
            "min_variance_unit": f"{units}^2",
            "max_variance": float(self.max_variance),
            "max_variance_unit": f"{units}^2",
            "noise_scale": float(self.noise_scale),
            "noise_scale_unit": "dimensionless_sigma_multiplier",
            "shot_noise_scale": self.shot_noise_scale,
            "shot_noise_scale_unit": f"electrons_per_{units}_of_declared_signal",
            "read_noise_signal": float(self.read_noise_signal),
            "read_noise_signal_unit": units,
            "normalisation": "none_in_sensor_unit_affine_is_declared_by_factors",
            "density_verified": bool(self.density_verified),
        }

    def sensor_variance(self, mean: Any) -> torch.Tensor:
        """Analytic sensor variance in ``units**2`` at a mean declared in ``units``.

        ``variance = clamp(mean, 0) / shot_noise_scale + read_noise_signal**2``, all in
        the declared units: no normalisation happens here.  For a declared unit with a
        black offset the caller passes the signal value the declaration was written
        against (see the class docstring and the factors-side ``UnitAffine``).
        """
        mean_tensor = _as_tensor(mean)
        if not mean_tensor.is_floating_point():
            mean_tensor = mean_tensor.to(torch.get_default_dtype())
        if self.shot_noise_scale is None and self.read_noise_signal == 0.0:
            return torch.zeros_like(mean_tensor)
        read_variance = float(self.read_noise_signal) ** 2
        if self.shot_noise_scale is None:
            variance = torch.full_like(mean_tensor, read_variance)
        else:
            variance = mean_tensor.clamp(min=0.0) / float(self.shot_noise_scale) + read_variance
        return (float(self.noise_scale) ** 2) * variance

    def predictive_variance(self, mean: Any, model_variance: Any) -> torch.Tensor:
        """Sensor variance plus the model's positive predictive uncertainty."""
        return predictive_variance(
            self.sensor_variance(mean),
            model_variance,
            min_variance=self.min_variance,
            max_variance=self.max_variance,
        )

    def build_prediction(
        self,
        mean: Any,
        *,
        model_variance: Any | None = None,
        variance_parameter: Any | None = None,
        parameterisation: str = "softplus",
        masks: Mapping[str, Any] | None = None,
        units: str | None = None,
    ) -> RawPrediction:
        """Assemble a :class:`RawPrediction` with the positive variance split recorded.

        ``mean`` and ``model_variance``/``variance_parameter`` are declared in
        ``self.units`` (variance in ``units**2``); the model's predictive uncertainty is
        added to the analytic sensor variance, never mixed across unit domains.
        """
        if model_variance is not None and variance_parameter is not None:
            raise ValueError("pass either model_variance or variance_parameter, not both")
        resolved_units = self.units if units is None else str(units)
        if resolved_units != self.units:
            raise UnitsMismatchError(
                f"prediction units {resolved_units!r} != model units {self.units!r}"
            )
        mean_tensor = _as_tensor(mean)
        if variance_parameter is not None:
            model_component = positive_variance(
                variance_parameter,
                parameterisation=parameterisation,
                min_variance=self.min_variance,
                max_variance=self.max_variance,
            )
        elif model_variance is not None:
            model_component = sanitise_variance(
                model_variance, min_variance=self.min_variance, max_variance=self.max_variance
            )
        else:
            model_component = torch.zeros_like(mean_tensor)
        sensor_component = self.sensor_variance(mean_tensor)
        total = predictive_variance(
            sensor_component,
            model_component,
            min_variance=self.min_variance,
            max_variance=self.max_variance,
        )
        declared_masks: dict[str, Any] = {}
        if masks:
            declared_masks.update(masks)
        declared_masks.setdefault("valid", torch.ones_like(mean_tensor, dtype=torch.bool))
        for name in ("bad", "missing", "saturated"):
            declared_masks.setdefault(name, torch.zeros_like(mean_tensor, dtype=torch.bool))
        return RawPrediction(
            mean=mean_tensor,
            variance=total,
            masks=declared_masks,
            units=resolved_units,
            variance_components={
                "sensor": sensor_component,
                "model": model_component,
                "total": total,
            },
        )

    def score(
        self,
        prediction: RawPrediction,
        values: Any,
        *,
        kind: str | None = None,
        units: str | None = None,
        **overrides: Any,
    ) -> MeasurementScore:
        """Score with this model's declared defaults; ``overrides`` win per call.

        ``units`` is the query's declared unit string; it is never inferred for a bare
        tensor, because an undeclared unit cannot be compared verbatim.  A query object
        carrying ``unit``/``units`` is accepted without repeating it.
        """
        if not isinstance(prediction, RawPrediction):
            raise TypeError(f"prediction must be a RawPrediction, got {type(prediction).__name__}")
        if prediction.units != self.units:
            raise UnitsMismatchError(
                f"prediction units {prediction.units!r} != model units {self.units!r}"
            )
        score_kind = self.kind if kind is None else kind
        if score_kind == "proper_nll" and not self.density_verified:
            raise DensityVerificationError(
                "the joint density of this sensor model is not verified; "
                "score it as empirical_energy"
            )
        declared: dict[str, Any] = {
            "quantisation_step": self.quantisation_step,
            "saturation_level": self.saturation_level,
            "floor_level": self.floor_level,
            "min_variance": self.min_variance,
            "max_variance": self.max_variance,
        }
        declared.update(overrides)
        return score_prediction(prediction, values, kind=score_kind, units=units, **declared)

    def diagnostics(
        self,
        prediction: RawPrediction,
        values: Any,
        *,
        units: str | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Score and summarise the magnitude for the run manifest (see
        :func:`score_diagnostics`); ``scale_suspect`` flags a silent unit/scale mistake."""
        return score_diagnostics(
            self.score(prediction, values, units=units, **overrides), prediction
        )


# --------------------------------------------------------------------------------------
# interference eta: history estimation and projection semantics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InterferenceObservation:
    """One history-declared interference record.

    Every field is declared by the history (camera metadata, odometry, a fitted noise
    scale).  It carries no truth field, no label and no future measurement.
    """

    frame_index: int
    camera_id: str
    log_exposure: float = 0.0
    camera_translation: tuple[float, float] = (0.0, 0.0)
    noise_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "camera_id", str(self.camera_id))
        translation = tuple(float(item) for item in self.camera_translation)
        if len(translation) != 2:
            raise ValueError("camera_translation must have two entries")
        object.__setattr__(self, "camera_translation", translation)
        log_exposure = float(self.log_exposure)
        noise_scale = float(self.noise_scale)
        if not (math.isfinite(log_exposure) and math.isfinite(noise_scale)):
            raise ValueError("log_exposure and noise_scale must be finite")
        if not noise_scale > 0.0:
            raise ValueError("noise_scale must be positive")
        if not all(math.isfinite(item) for item in translation):
            raise ValueError("camera_translation entries must be finite")
        object.__setattr__(self, "log_exposure", log_exposure)
        object.__setattr__(self, "noise_scale", noise_scale)


@dataclass(frozen=True, slots=True)
class InterferencePrior:
    """Development-set frozen range and prior for eta (P1 section 3)."""

    log_exposure_mean: float = 0.0
    log_exposure_std: float = 1.0
    log_exposure_observation_std: float = 0.5
    log_exposure_range: tuple[float, float] = (-4.0, 4.0)
    translation_range_px: tuple[float, float] = (-24.0, 24.0)
    noise_scale_prior: float = 1.0
    noise_scale_range: tuple[float, float] = (0.25, 4.0)

    def __post_init__(self) -> None:
        if not self.log_exposure_std > 0.0 or not self.log_exposure_observation_std > 0.0:
            raise ValueError("log-exposure standard deviations must be positive")
        low, high = (float(item) for item in self.log_exposure_range)
        if not low <= self.log_exposure_mean <= high:
            raise ValueError("log_exposure_mean must lie inside log_exposure_range")
        object.__setattr__(self, "log_exposure_range", (low, high))
        low, high = (float(item) for item in self.translation_range_px)
        if not low <= 0.0 <= high:
            raise ValueError("translation_range_px must contain zero")
        object.__setattr__(self, "translation_range_px", (low, high))
        low, high = (float(item) for item in self.noise_scale_range)
        if not low <= self.noise_scale_prior <= high:
            raise ValueError("noise_scale_prior must lie inside noise_scale_range")
        object.__setattr__(self, "noise_scale_range", (low, high))


#: The frozen P1 development-set default prior; callers may pass their own frozen prior.
DEFAULT_INTERFERENCE_PRIOR = InterferencePrior()


def _forbidden_attribute_names(item: Any) -> list[str]:
    if isinstance(item, Mapping):
        names = {str(key) for key in item}
    elif is_dataclass(item) and not isinstance(item, type):
        names = {member.name for member in fields(item)}
    elif hasattr(item, "__dict__"):
        names = set(vars(item))
    else:
        names = set()
    return sorted(
        name for name in names if any(token in name.lower() for token in _FORBIDDEN_TOKENS)
    )


def _reject_evaluator_inputs(records: Sequence[Any]) -> None:
    for item in records:
        type_name = type(item).__name__.lower()
        if any(token in type_name for token in ("truth", "oracle", "evaluation")):
            raise ValueError(
                "eta must be estimated from history only; refused evaluator-side object "
                f"{type(item).__name__}"
            )
        leaked = _forbidden_attribute_names(item)
        if leaked:
            raise ValueError(f"eta history input carries forbidden evaluator fields: {leaked}")


def estimate_interference(
    history: Sequence[InterferenceObservation],
    *,
    prior: InterferencePrior = DEFAULT_INTERFERENCE_PRIOR,
    frame_indices: Sequence[int] | None = None,
    camera_ids: Sequence[str] | None = None,
) -> InterferenceEstimate:
    """Estimate eta from history-declared records only.

    The shared log exposure shrinks toward the frozen prior, per ``(frame, camera)``
    translation is the history mean clamped to the frozen range, and the noise scale is
    the geometric mean clamped to the frozen range.  An empty history returns the prior
    itself.  The result is always ``source="history"``.

    The estimator refuses evaluator-side objects (a type or field naming truth/oracle/
    evaluation/label/future/target), so it cannot read eta from the evaluation truth.
    """
    records = tuple(history)
    _reject_evaluator_inputs(records)
    for record in records:
        if not isinstance(record, InterferenceObservation):
            raise TypeError(
                f"history records must be InterferenceObservation, got {type(record).__name__}"
            )

    prior_precision = 1.0 / float(prior.log_exposure_std) ** 2
    observation_precision = 1.0 / float(prior.log_exposure_observation_std) ** 2
    numerator = float(prior.log_exposure_mean) * prior_precision
    for record in records:
        numerator += float(record.log_exposure) * observation_precision
    denominator = prior_precision + len(records) * observation_precision
    log_exposure = numerator / denominator
    low, high = prior.log_exposure_range
    log_exposure = min(max(log_exposure, low), high)

    resolved_frames = (
        tuple(sorted({int(item) for item in frame_indices}))
        if frame_indices is not None
        else tuple(sorted({record.frame_index for record in records}))
    )
    resolved_cameras = (
        tuple(str(item) for item in camera_ids)
        if camera_ids is not None
        else tuple(sorted({record.camera_id for record in records}))
    )
    if not resolved_frames:
        resolved_frames = (0,)
    if not resolved_cameras:
        resolved_cameras = ("camera0",)
    frame_position = {frame: index for index, frame in enumerate(resolved_frames)}
    camera_position = {camera: index for index, camera in enumerate(resolved_cameras)}
    shift_sum = torch.zeros((len(resolved_frames), len(resolved_cameras), 2), dtype=torch.float64)
    shift_count = torch.zeros((len(resolved_frames), len(resolved_cameras), 1), dtype=torch.float64)
    for record in records:
        if record.frame_index not in frame_position or record.camera_id not in camera_position:
            continue
        row = frame_position[record.frame_index]
        column = camera_position[record.camera_id]
        shift_sum[row, column] += torch.tensor(record.camera_translation, dtype=torch.float64)
        shift_count[row, column] += 1.0
    translation = shift_sum / shift_count.clamp_min(1.0)
    bound_low, bound_high = prior.translation_range_px
    translation = translation.clamp(min=bound_low, max=bound_high)

    if records:
        log_noise = sum(math.log(float(record.noise_scale)) for record in records) / len(records)
        noise_scale = math.exp(log_noise)
    else:
        noise_scale = float(prior.noise_scale_prior)
    noise_low, noise_high = prior.noise_scale_range
    noise_scale = min(max(noise_scale, noise_low), noise_high)

    return InterferenceEstimate(
        log_exposure=torch.tensor(log_exposure, dtype=torch.float64),
        camera_translation=translation,
        noise_scale=torch.tensor(noise_scale, dtype=torch.float64),
        source="history",
    )


def project_camera_motion(coordinates: Any, translation: Any) -> torch.Tensor:
    """Camera motion shifts every projected point (background and objects alike)."""
    coords = _as_tensor(coordinates)
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    shift = _as_tensor(translation).to(coords.dtype).reshape(-1)
    if shift.numel() != 2:
        raise ValueError("translation must have two entries (dy, dx)")
    return coords + shift


def project_entity_motion(
    coordinates: Any,
    translation: Any,
    *,
    entity_mask: Any,
) -> torch.Tensor:
    """Entity motion shifts only that entity's own projected points."""
    coords = _as_tensor(coordinates)
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    shift = _as_tensor(translation).to(coords.dtype).reshape(-1)
    if shift.numel() != 2:
        raise ValueError("translation must have two entries (dy, dx)")
    mask = _as_tensor(entity_mask, dtype=torch.bool)
    if tuple(mask.shape) != tuple(coords.shape[:-1]):
        raise ValueError(
            f"entity_mask shape {tuple(mask.shape)} must match coordinates without the last "
            f"axis {tuple(coords.shape[:-1])}"
        )
    shifted = coords + shift
    return torch.where(mask.unsqueeze(-1), shifted, coords)


# --------------------------------------------------------------------------------------
# declared probabilities re-used from the synthetic model (one implementation, no copy)
# --------------------------------------------------------------------------------------


def declared_saturation_tail_probability(mean: Any, variance: Any, *, level: float) -> np.ndarray:
    """The declared numpy ``P(noisy >= level)`` of the synthetic measurement model."""
    return saturation_tail_probability(mean, variance, level)


def declared_quantisation_interval_probability(
    value: Any, mean: Any, variance: Any, *, step: float = 1.0
) -> np.ndarray:
    """The declared numpy ``P(value - step/2 <= noisy < value + step/2)``."""
    return quantisation_interval_probability(value, mean, variance, step=step)

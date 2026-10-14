"""First-version sensor noise for N01 v1.0 P2: shot/read plus a shared black level.

The declared observation model, in the RAW unit of the values it is applied to, is

    y = mu + e_block + e_read
    Var(y | mu) = shot * max(mu - black_level, 0) + read_var
    e_block ~ N(0, black_level.variance) shared by every sample of one noise block

``shot``/``read_var`` are numbers in the *same* unit as ``mu``: they are never
rescaled silently.  Exposure and gain change value and variance together
(``apply_exposure_gain``), so a rescaled parameter set is a new object with its
own provenance instead of an edited one.

Two provenances are kept apart on purpose and both are mandatory in JSON:

* ``explicit_synthesis`` -- a number this project chose to synthesise data with;
* ``fitted`` -- a number obtained from data, and only from the train or
  calibration region.  Fitting on the test region is refused here, and the
  validation region is reserved for checkpoint selection, not for fitting.

:func:`interpolated_variance` evaluates ``a_taps.T @ Sigma @ a_taps`` with the
real tap weights.  Interpolating a *variance map* is a different (wrong)
quantity and is not implemented here; the test module keeps that counterexample.

:func:`flow_first_order_variance` propagates a *flow* covariance through the
image gradient ``g.T @ Sigma_flow @ g``.  It is a first-order linearisation: it
is invalid across strong edges and occlusion discontinuities, where the image is
not locally affine.  It is a diagnostic tool, never sensor noise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from owac.frameworks.n01_full.contracts import SourceKey
from owac.frameworks.n01_full.statistics import NoiseModel as StatisticsNoiseModel

__all__ = [
    "CORRELATION_KINDS",
    "CROSS_PHASE_UNKNOWN_REFUSAL",
    "FITTING_SPLITS",
    "MAX_DENSE_SOURCES",
    "NOISE_FORMULA",
    "NOISE_SCHEMA",
    "NOISE_VERSION_PREFIX",
    "PROVENANCE_KINDS",
    "PROVENANCE_SPLITS",
    "NoiseCalibration",
    "NoiseProvenance",
    "SharedBlackLevel",
    "ShotReadParameters",
    "UnsupportedNoiseStructureError",
    "add_sensor_noise",
    "apply_exposure_gain",
    "build_noise_model",
    "draw_block_error",
    "flow_first_order_variance",
    "interpolated_variance",
    "load_noise_calibration",
    "shot_read_variance",
]

NOISE_SCHEMA = "owac.n01_full.noise.v1"
NOISE_FORMULA = "Var(y | mu) = shot * max(mu - black_level, 0) + read_var"
NOISE_VERSION_PREFIX = "n01-noise-"

#: ``explicit_synthesis`` numbers are chosen; ``fitted`` numbers come from data.
PROVENANCE_KINDS = ("explicit_synthesis", "fitted")
#: The four data regions of the shared contract plus the synthesis pseudo-region.
PROVENANCE_SPLITS = ("synthesis", "train", "calibration", "validation", "test")
#: Only these regions may fit a noise parameter.
FITTING_SPLITS = ("train", "calibration")
#: How the shared black-level correlation is declared to ``build_noise_model``.
CORRELATION_KINDS = ("declared", "unknown")

#: A dense joint covariance is materialised only for the small quotient set that
#: enters one fit (taps of a row or a phase-local neighbourhood).  Larger sets must
#: keep the block as a factor instead of a dense matrix.
MAX_DENSE_SOURCES = 256

#: The one structure the existing ``statistics.NOISE_KINDS`` cannot express exactly.
CROSS_PHASE_UNKNOWN_REFUSAL = (
    "a shared black-level block with an unaudited magnitude cannot be written as an exact "
    "diagonal or dense per-sample covariance; build_noise_model(correlation='declared') "
    "refuses it rather than dropping the shared block, and correlation='unknown' returns the "
    "proxy-labelled unknown_correlation kind instead"
)


class UnsupportedNoiseStructureError(ValueError):
    """The declared noise structure has no exact representation in ``NOISE_KINDS``."""


def _reject_unknown_keys(data: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    """A declared JSON vocabulary is closed: an unknown field is an error, not a comment."""
    extra = sorted(set(data) - set(allowed))
    if extra:
        raise ValueError(
            f"{what} carries unknown fields {extra}; the declared vocabulary is {sorted(allowed)}"
        )


def noise_version(calibration_version: str) -> str:
    """Version string a ``NoiseModel`` carries, and a domain binds via ``noise_version``."""
    if not isinstance(calibration_version, str) or not calibration_version:
        raise ValueError("calibration_version must be a non-empty string")
    return f"{NOISE_VERSION_PREFIX}{calibration_version}"


@dataclass(frozen=True)
class NoiseProvenance:
    """Where one declared number came from, in the four-region vocabulary.

    ``kind='explicit_synthesis'`` is a chosen number and lives in the synthesis
    pseudo-region; ``kind='fitted'`` is an estimate and may only come from
    ``train`` or ``calibration``.  ``parameters_known`` records whether the
    magnitude is audited at all: an unknown parameter is stored as unknown, not
    as a zero.
    """

    kind: str
    split: str
    source: str
    method: str
    parameters_known: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PROVENANCE_KINDS:
            raise ValueError(f"unknown provenance kind {self.kind!r}; expected {PROVENANCE_KINDS}")
        if self.split not in PROVENANCE_SPLITS:
            raise ValueError(
                f"unknown provenance split {self.split!r}; expected {PROVENANCE_SPLITS}"
            )
        if self.kind == "explicit_synthesis" and self.split != "synthesis":
            raise ValueError(
                "an explicitly synthesised parameter has no data region; declare "
                f"split='synthesis', got {self.split!r}"
            )
        if self.kind == "fitted" and self.split not in FITTING_SPLITS:
            raise ValueError(
                "a fitted noise parameter is fitted on the train or calibration region only; "
                f"got split {self.split!r}.  The test region is never a fitting input and the "
                "validation region selects checkpoints."
            )
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("NoiseProvenance.source must name the fitted or synthesised source")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("NoiseProvenance.method must describe how the number was obtained")
        if not isinstance(self.parameters_known, bool):
            raise TypeError("NoiseProvenance.parameters_known must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "split": self.split,
            "source": self.source,
            "method": self.method,
            "parameters_known": self.parameters_known,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoiseProvenance:
        _reject_unknown_keys(
            data,
            ("kind", "split", "source", "method", "parameters_known", "notes"),
            "NoiseProvenance",
        )
        return cls(
            kind=str(data["kind"]),
            split=str(data["split"]),
            source=str(data["source"]),
            method=str(data["method"]),
            parameters_known=bool(data.get("parameters_known", True)),
            notes=str(data.get("notes", "")),
        )


def shot_read_variance(
    mean: Any, *, shot: float, read_var: float, black_level: float = 0.0
) -> torch.Tensor:
    """``shot * max(mean - black_level, 0) + read_var`` in the unit of ``mean``.

    The parameters are used exactly as given: a caller working in another unit
    must rescale them first (see :meth:`ShotReadParameters.scaled`).
    """
    shot = float(shot)
    read_var = float(read_var)
    black = float(black_level)
    if shot < 0.0 or read_var < 0.0:
        raise ValueError(f"shot and read_var must be non-negative, got {shot} and {read_var}")
    values = torch.as_tensor(mean, dtype=torch.float64)
    signal = torch.clamp(values - black, min=0.0)
    return shot * signal + read_var


@dataclass(frozen=True)
class ShotReadParameters:
    """``a`` (shot slope) and ``b`` (read variance) of one RAW unit."""

    shot: float
    read_var: float
    units: str
    provenance: NoiseProvenance

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.shot)) or float(self.shot) < 0.0:
            raise ValueError(f"shot must be finite and non-negative, got {self.shot!r}")
        if not math.isfinite(float(self.read_var)) or float(self.read_var) <= 0.0:
            raise ValueError(
                "read_var must be finite and strictly positive so that every declared "
                f"covariance is non-singular; got {self.read_var!r}"
            )
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("ShotReadParameters.units must name the RAW unit")
        if not isinstance(self.provenance, NoiseProvenance):
            raise TypeError("ShotReadParameters.provenance must be a NoiseProvenance")

    def variance(self, mean: Any, *, black_level: float = 0.0) -> torch.Tensor:
        return shot_read_variance(
            mean, shot=self.shot, read_var=self.read_var, black_level=black_level
        )

    def scaled(
        self, gain: float, *, provenance: NoiseProvenance | None = None
    ) -> ShotReadParameters:
        """Rescale the parameters for a measurement multiplied by ``gain``.

        For ``mu' = black + gain * (mu - black)`` and ``Var' = gain**2 * Var`` the
        law keeps its form with ``shot' = shot * gain`` and ``read_var' = gain**2
        * read_var``, so value and variance stay consistent under one gain.
        """
        gain = float(gain)
        if not math.isfinite(gain) or gain <= 0.0:
            raise ValueError(f"gain must be finite and positive, got {gain!r}")
        derived = provenance
        if derived is None:
            derived = replace(
                self.provenance,
                method=(
                    f"{self.provenance.method}; then rescaled by gain={gain!r} "
                    "(shot'=shot*gain, read_var'=read_var*gain**2)"
                ),
            )
        return ShotReadParameters(
            shot=self.shot * gain,
            read_var=self.read_var * gain * gain,
            units=self.units,
            provenance=derived,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot": float(self.shot),
            "read_var": float(self.read_var),
            "units": self.units,
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShotReadParameters:
        _reject_unknown_keys(
            data, ("shot", "read_var", "units", "provenance"), "ShotReadParameters"
        )
        return cls(
            shot=float(data["shot"]),
            read_var=float(data["read_var"]),
            units=str(data["units"]),
            provenance=NoiseProvenance.from_dict(data["provenance"]),
        )


@dataclass(frozen=True)
class SharedBlackLevel:
    """One black-level error shared by every sample of a ``noise_block_id``.

    ``variance`` is the audited ``sigma_b**2``; ``None`` means the magnitude is
    explicitly unknown, in which case ``upper_bound`` must state the audit bound
    that stands in for it.  ``id_template`` names the block from a
    :class:`SourceKey`, so a block is per frame/episode/camera as declared and a
    re-delivered frame keeps the same block.
    """

    kind: str
    id_template: str
    variance: float | None
    provenance: NoiseProvenance
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("SharedBlackLevel.kind must name the block grouping rule")
        if not isinstance(self.id_template, str) or "{frame_id}" not in self.id_template:
            raise ValueError(
                "SharedBlackLevel.id_template must name a block from SourceKey fields and "
                f"include '{{frame_id}}'; got {self.id_template!r}"
            )
        if self.variance is not None:
            if not math.isfinite(float(self.variance)) or float(self.variance) <= 0.0:
                raise ValueError(
                    f"a declared block variance must be finite and positive, got {self.variance!r}"
                )
        if self.upper_bound is not None:
            if not math.isfinite(float(self.upper_bound)) or float(self.upper_bound) < 0.0:
                raise ValueError(
                    "a declared upper bound must be finite and non-negative, got "
                    f"{self.upper_bound!r}"
                )
        if self.variance is None and self.upper_bound is None:
            raise ValueError(
                "an unaudited shared block is recorded as unknown, not as zero: declare "
                "upper_bound (or variance) instead of leaving both empty"
            )
        if not isinstance(self.provenance, NoiseProvenance):
            raise TypeError("SharedBlackLevel.provenance must be a NoiseProvenance")

    @property
    def audited(self) -> bool:
        """Whether the block magnitude itself is a known number."""
        return self.variance is not None

    def block_key(self, key: SourceKey) -> str:
        """Concrete ``noise_block_id`` of this sample under the declared template."""
        if not isinstance(key, SourceKey):
            raise TypeError("SharedBlackLevel.block_key expects a SourceKey")
        return self.id_template.format(
            dataset_id=key.dataset_id,
            episode_id=key.episode_id,
            camera_id=key.camera_id,
            frame_id=key.frame_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id_template": self.id_template,
            "variance": None if self.variance is None else float(self.variance),
            "upper_bound": None if self.upper_bound is None else float(self.upper_bound),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SharedBlackLevel:
        _reject_unknown_keys(
            data,
            ("kind", "id_template", "variance", "upper_bound", "provenance"),
            "SharedBlackLevel",
        )
        variance = data.get("variance")
        upper_bound = data.get("upper_bound")
        return cls(
            kind=str(data["kind"]),
            id_template=str(data["id_template"]),
            variance=None if variance is None else float(variance),
            upper_bound=None if upper_bound is None else float(upper_bound),
            provenance=NoiseProvenance.from_dict(data["provenance"]),
        )


@dataclass(frozen=True)
class NoiseCalibration:
    """A versioned, loadable noise declaration with per-number provenance.

    ``refused_structures`` is part of the artifact, not a comment: it records
    which noise structures this version cannot express exactly and which
    substitute label was used instead.
    """

    calibration_version: str
    units: str
    shot_read: ShotReadParameters
    black_level: SharedBlackLevel
    black_level_dn: float = 0.0
    white_level_dn: float | None = None
    refused_structures: tuple[str, ...] = ()
    notes: str = ""
    schema: str = NOISE_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_version, str) or not self.calibration_version:
            raise ValueError("NoiseCalibration.calibration_version must be non-empty")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("NoiseCalibration.units must name the RAW unit")
        if self.shot_read.units != self.units:
            raise ValueError(
                f"shot/read units {self.shot_read.units!r} differ from the declared units "
                f"{self.units!r}; a unit mismatch is not a rounding detail"
            )
        if self.schema != NOISE_SCHEMA:
            raise ValueError(f"unknown noise schema {self.schema!r}; expected {NOISE_SCHEMA!r}")
        if not math.isfinite(float(self.black_level_dn)):
            raise ValueError("black_level_dn must be finite")
        if self.white_level_dn is not None:
            if not math.isfinite(float(self.white_level_dn)):
                raise ValueError("white_level_dn must be finite when declared")
            if float(self.white_level_dn) <= float(self.black_level_dn):
                raise ValueError("white_level_dn must exceed black_level_dn")
        if not all(isinstance(item, str) and item for item in self.refused_structures):
            raise ValueError("refused_structures entries must be non-empty strings")
        object.__setattr__(self, "refused_structures", tuple(self.refused_structures))

    @property
    def noise_version(self) -> str:
        """The version a ``statistics.NoiseModel`` and a domain binding must agree on."""
        return noise_version(self.calibration_version)

    def block_key(self, key: SourceKey) -> str:
        return self.black_level.block_key(key)

    def variance(self, mean: Any, *, black_level: float | None = None) -> torch.Tensor:
        """Per-sample variance including an audited block term, in the declared unit."""
        black = self.black_level_dn if black_level is None else float(black_level)
        variances = self.shot_read.variance(mean, black_level=black)
        if self.black_level.audited:
            variances = variances + float(self.black_level.variance)
        return variances

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "calibration_version": self.calibration_version,
            "noise_version": self.noise_version,
            "units": self.units,
            "formula": NOISE_FORMULA,
            "shot_read": self.shot_read.to_dict(),
            "black_level": self.black_level.to_dict(),
            "black_level_dn": float(self.black_level_dn),
            "white_level_dn": None if self.white_level_dn is None else float(self.white_level_dn),
            "refused_structures": list(self.refused_structures),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NoiseCalibration:
        _reject_unknown_keys(
            data,
            (
                "schema",
                "calibration_version",
                "noise_version",
                "units",
                "formula",
                "shot_read",
                "black_level",
                "black_level_dn",
                "white_level_dn",
                "refused_structures",
                "notes",
            ),
            "NoiseCalibration",
        )
        for field in (
            "schema",
            "calibration_version",
            "units",
            "shot_read",
            "black_level",
            "black_level_dn",
        ):
            if field not in data:
                raise ValueError(f"noise calibration JSON is missing the required field {field!r}")
        white = data.get("white_level_dn")
        calibration = cls(
            schema=str(data["schema"]),
            calibration_version=str(data["calibration_version"]),
            units=str(data["units"]),
            shot_read=ShotReadParameters.from_dict(data["shot_read"]),
            black_level=SharedBlackLevel.from_dict(data["black_level"]),
            black_level_dn=float(data["black_level_dn"]),
            white_level_dn=None if white is None else float(white),
            refused_structures=tuple(str(item) for item in data.get("refused_structures", ())),
            notes=str(data.get("notes", "")),
        )
        declared = data.get("noise_version")
        if declared is not None and str(declared) != calibration.noise_version:
            raise ValueError(
                f"declared noise_version {declared!r} does not match the version derived from "
                f"calibration_version {calibration.calibration_version!r} "
                f"({calibration.noise_version!r}); a domain binding would check the derived one"
            )
        return calibration

    @classmethod
    def from_path(cls, path: str | Path) -> NoiseCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return cls.from_dict(payload)

    def to_path(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


def load_noise_calibration(path: str | Path) -> NoiseCalibration:
    """Load and validate a calibration JSON; provenance rules are enforced on load."""
    return NoiseCalibration.from_path(path)


def build_noise_model(
    source_order: Sequence[SourceKey],
    means: Sequence[float],
    calibration: NoiseCalibration,
    *,
    block_keys: Sequence[str | None] | None = None,
    black_level: float | None = None,
    correlation: str = "declared",
    version: str | None = None,
) -> StatisticsNoiseModel:
    """Build the P1 ``statistics.NoiseModel`` for exactly these observations.

    ``means`` are the exposure-scaled RAW means of the sources in
    ``source_order``.  ``block_keys`` names the shared black-level block of each
    sample (default: derived from the :class:`SourceKey` through the declared
    template); ``None`` marks a sample that shares no block.

    Every sample that belongs to a block carries the block marginal in addition
    to its own shot/read variance, in every branch: the declared marginal equals
    :meth:`NoiseCalibration.variance`, so it never depends on which sibling
    samples happen to enter this fit.  ``correlation='declared'`` then builds the
    exact model -- ``diagonal`` when no two samples share a block, otherwise
    ``dense`` with the block as a rank-one term::

        Sigma = diag(shot * max(mu - black, 0) + read_var)
                + sum_groups sigma_b**2 * 1_g 1_g.T

    The dense matrix stays positive definite because every sample keeps a
    strictly positive read variance, and its diagonal equals the diagonal
    branch's variances for the same sample set (singleton blocks included).

    ``correlation='unknown'`` is the honest fallback when the block magnitude is
    unaudited: the marginals are the declared per-sample variance plus the
    declared ``upper_bound`` of the unaudited block, the kind label stays the
    proxy ``unknown_correlation``, and the correlation itself is handled by that
    kind's weight bound rather than by an exact covariance.  With
    ``correlation='declared'`` an unaudited block raises
    :class:`UnsupportedNoiseStructureError` instead of silently dropping it.
    """
    order = tuple(source_order)
    if not order:
        raise ValueError("build_noise_model needs at least one source")
    for key in order:
        if not isinstance(key, SourceKey):
            raise TypeError("build_noise_model expects SourceKey entries in source_order")
    values = [float(mean) for mean in means]
    if len(values) != len(order):
        raise ValueError(
            f"means has {len(values)} entries for {len(order)} declared sources; a positional "
            "or length mismatch would mis-assign variances"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("every declared mean must be finite")
    black = calibration.black_level_dn if black_level is None else float(black_level)
    keys = (
        tuple(calibration.block_key(key) for key in order)
        if block_keys is None
        else tuple(block_keys)
    )
    if len(keys) != len(order):
        raise ValueError("block_keys must align with source_order")
    for key in keys:
        if key is not None and (not isinstance(key, str) or not key):
            raise ValueError(f"a block key must be a non-empty string or None, got {key!r}")

    if correlation not in CORRELATION_KINDS:
        raise ValueError(f"unknown correlation {correlation!r}; expected {CORRELATION_KINDS}")

    base_variances = [
        float(item) for item in calibration.shot_read.variance(values, black_level=black)
    ]
    block = calibration.black_level
    model_version = calibration.noise_version if version is None else str(version)
    if not model_version:
        raise ValueError("a NoiseModel version must be declared")

    # One marginal rule for every branch: a sample that belongs to a block carries the
    # audited ``sigma_b**2`` (or, while the magnitude is unaudited, the declared upper
    # bound) *in addition to* its own shot/read variance.  Dropping it because no other
    # sample of the same block happens to sit in this fit would make one sample's declared
    # noise depend on which siblings were passed in.
    marginal_block = float(block.variance) if block.audited else float(block.upper_bound)
    block_terms = [0.0 if key is None else marginal_block for key in keys]

    if correlation == "unknown":
        # The proxy label is unchanged: the marginals are the declared per-sample
        # variance and the kind's own weight bound handles the unknown correlation.
        variances = [base + term for base, term in zip(base_variances, block_terms, strict=True)]
        return StatisticsNoiseModel.unknown_correlation(
            variances, source_order=order, version=model_version
        )

    if not block.audited:
        raise UnsupportedNoiseStructureError(
            f"{CROSS_PHASE_UNKNOWN_REFUSAL}; declared block {block.kind!r} for "
            f"{len(set(keys))} block(s) of {len(order)} sources"
        )
    groups: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        if key is not None:
            groups.setdefault(key, []).append(position)
    shared = {key: positions for key, positions in groups.items() if len(positions) > 1}
    if not shared:
        return StatisticsNoiseModel.diagonal(
            [base + term for base, term in zip(base_variances, block_terms, strict=True)],
            source_order=order,
            version=model_version,
        )
    if len(order) > MAX_DENSE_SOURCES:
        raise UnsupportedNoiseStructureError(
            f"a dense joint covariance over {len(order)} sources exceeds the declared limit "
            f"{MAX_DENSE_SOURCES}; keep the shared block as a factor for the full frame and "
            "build the dense model only for the quotient set entering one fit"
        )
    # ``diag(shot/read) + sum_groups sigma_b**2 * 1_g 1_g.T`` carries the block marginal on
    # the diagonal and the within-block correlation off it, singletons included, so the
    # diagonal entries equal the diagonal branch's variances for the same sample set.
    matrix = torch.diag(torch.tensor(base_variances, dtype=torch.float64))
    for positions in groups.values():
        index = torch.tensor(positions, dtype=torch.long)
        matrix[index.unsqueeze(1), index.unsqueeze(0)] += float(block.variance)
    return StatisticsNoiseModel.dense(order, matrix, version=model_version)


def draw_block_error(
    block_id: str, calibration: NoiseCalibration, generator: torch.Generator
) -> float:
    """Draw the shared black-level error of one block; refused when unaudited."""
    block = calibration.black_level
    if not block.audited:
        raise UnsupportedNoiseStructureError(
            f"{CROSS_PHASE_UNKNOWN_REFUSAL}; block {block_id!r} has no audited magnitude to draw"
        )
    if not isinstance(block_id, str) or not block_id:
        raise ValueError("block_id must be the non-empty identity of the drawn block")
    if not isinstance(generator, torch.Generator):
        raise TypeError("draw_block_error needs an explicit torch.Generator for reproducibility")
    sigma = math.sqrt(float(block.variance))
    draw = torch.randn((), dtype=torch.float64, generator=generator)
    return float(draw) * sigma


def add_sensor_noise(
    clean_dn: Any,
    *,
    calibration: NoiseCalibration,
    generator: torch.Generator,
    block_error: float = 0.0,
    black_level: float | None = None,
) -> torch.Tensor:
    """Add one drawn noise realisation: shared block error plus per-sample read noise.

    The realisation is computed in FP64 and returned in FP64; the caller casts
    to the storage dtype so a cast never feeds back into the draw.
    """
    if not isinstance(generator, torch.Generator):
        raise TypeError("add_sensor_noise needs an explicit torch.Generator for reproducibility")
    black = calibration.black_level_dn if black_level is None else float(black_level)
    mu = torch.as_tensor(clean_dn, dtype=torch.float64)
    variance = calibration.shot_read.variance(mu, black_level=black)
    noise = torch.randn(mu.shape, dtype=torch.float64, generator=generator)
    return mu + float(block_error) + noise * variance.sqrt()


def apply_exposure_gain(
    values: Any, variance: Any, *, gain: float, black_level: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale value and variance by one gain: ``v' = black + gain*(v-black)``, ``var' = g**2*var``.

    The black level is a sensor offset in DN and does not scale; the signal above
    it does.  A caller that rescales the value without the variance, or with the
    wrong power of the gain, gets a different pair and fails the numeric test.
    """
    gain = float(gain)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"gain must be finite and non-negative, got {gain!r}")
    scaled_values = float(black_level) + gain * (
        torch.as_tensor(values, dtype=torch.float64) - float(black_level)
    )
    scaled_variance = gain * gain * torch.as_tensor(variance, dtype=torch.float64)
    return scaled_values, scaled_variance


def interpolated_variance(tap_weights: Any, covariance: Any) -> torch.Tensor:
    """``a_taps.T @ Sigma @ a_taps`` for the *real* tap weights of one sampling row.

    ``tap_weights`` is ``[..., n_taps]`` and ``covariance`` is the ``[n_taps,
    n_taps]`` joint covariance of exactly those taps (for example the sub-block a
    ``statistics.NoiseModel`` returns for the row's ``tap_source_ids``).  A
    variance map interpolated by the same weights is a different quantity; the
    test module keeps the counterexample that must fail.
    """
    weights = torch.as_tensor(tap_weights, dtype=torch.float64)
    sigma = torch.as_tensor(covariance, dtype=torch.float64)
    if weights.ndim < 1:
        raise ValueError("tap_weights must carry at least one tap axis")
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"covariance must be square, got shape {tuple(sigma.shape)}")
    if int(weights.shape[-1]) != int(sigma.shape[0]):
        raise ValueError(
            f"tap_weights has {int(weights.shape[-1])} taps but the covariance covers "
            f"{int(sigma.shape[0])}; the weights must be the real taps of this row"
        )
    return torch.einsum("...i,ij,...j->...", weights, sigma, weights)


def flow_first_order_variance(image_gradient: Any, flow_covariance: Any) -> torch.Tensor:
    """``grad(f).T @ Sigma_flow @ grad(f)`` -- a first-order approximation only.

    This linearises the image around the current estimate.  Across a strong edge
    or an occlusion boundary the image is not locally affine, so the true
    propagated variance can exceed this value arbitrarily (and the gradient
    itself is undefined at a jump).  It is a diagnostic instrument for smooth
    regions, never an unconditional sensor-noise model, and it must not be added
    to a sensor variance as if the two were the same quantity.
    """
    gradient = torch.as_tensor(image_gradient, dtype=torch.float64)
    sigma = torch.as_tensor(flow_covariance, dtype=torch.float64)
    if gradient.shape[-1] != 2:
        raise ValueError(
            f"image_gradient must carry the [dx, dy] axis last, got {tuple(gradient.shape)}"
        )
    if sigma.ndim != 2 or tuple(sigma.shape) != (2, 2):
        raise ValueError(f"flow_covariance must be [2, 2], got {tuple(sigma.shape)}")
    return torch.einsum("...i,ij,...j->...", gradient, sigma, gradient)

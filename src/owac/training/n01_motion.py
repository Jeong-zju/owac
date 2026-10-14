"""P2 RAW motion training: author-style iterative loss, explicit units, platform receipts.

Loss and unit contract
----------------------
The vendored SEA-RAFT network predicts a **packed** displacement field
``[2, Hp, Wp]``: one packed cell step is ``sensor_per_packed`` sensor pixels
(``2`` for the ``cfa4`` stem, ``1`` for ``rgb3``).  Every flow target built here
is therefore expressed in packed cells of the *source* frame and converted to
sensor pixels only for reporting, through :func:`packed_to_sensor` /
:func:`sensor_to_packed`.  Mixing the two units is the failure mode the unit
tests pin down: a sensor-pixel target divided by the packed step is the packed
target, and a packed displacement multiplied by the step is the sensor one.

:func:`flow_target_from_supervision` builds the target from the generator's
analytic pull mapping.  It averages the per-channel packed displacement over the
channels whose supervision is ``valid`` and marks a cell valid only when at
least one channel is valid, so occluded (``occluded_in_source``) and
out-of-frame (``out_of_bounds``) cells never enter the loss.  The generator's
per-channel phase offsets are carried by the target itself, not assumed away.

:func:`sequence_loss` is the author's ``sequence_loss(output, flow_gt, valid,
gamma)`` (``core/loss.py`` at commit
``9137517ba24e628442aec097d3afe71d03503b75``, sha256
``12396473603867e32cd1f25ba9123adbcc4709430593df957dd97c24ca41e2fb``, fetched
through the GitHub contents API).  The vendored tree does not ship ``loss.py``
and this task owns only ``training/`` and ``evaluation/``, so the function is
re-implemented here with the same signature, the same ``gamma ** (n - i - 1)``
iteration weights, the same per-iteration masked mean and the same
``mag < max_flow`` exclusion.  Declared difference: the mask sum is clamped to
one so an empty mask yields a zero contribution instead of ``0/0`` (the trainer
refuses an all-invalid batch before the loss anyway).

Platform contract (shared contract section 6)
---------------------------------------------
Validation runs every ``validation_interval`` updates.  ``validation_plateau``
requires the last ``plateau_checks`` improvements of the monitored validation
metric to be below the locked ``delta_val``, at least two effective LR drops to
have been applied, the current phase to be the final (lowest) LR phase, the best
checkpoint not to sit at the budget end, and the small-set fit / input /
gradient diagnostics to pass.  It additionally requires a real improvement over
the run (``first - best > delta_val``, strictly) and a tail that has not drifted
away from its best (``|final - best| <= delta_val``): a constant curve and a
curve that only degrades are never plateaus.  Anything else that stops at the
budget is ``budget_exhausted``; the platform status is not a research verdict.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from owac.backends.raw_correspondence import (
    MODES,
    SENSOR_PER_PACKED,
    CorrespondenceEstimator,
    RawNormalization,
    RawSensorFrame,
)
from owac.raw.n01_motion_data import (
    FrameGeometry,
    MotionDataGenerator,
    MotionSupervision,
    P2MotionNoiseConfig,
    SceneRecipe,
    load_p2_motion_noise_config,
    parse_subpixel_token,
)
from owac.raw.n01_noise import apply_exposure_gain
from owac.training.n01 import git_sha

__all__ = [
    "AUTHOR_LOSS_COMMIT",
    "AUTHOR_LOSS_SHA256",
    "AUTHOR_TRAIN_CONFIG",
    "DEFAULT_GAMMA",
    "DEFAULT_MAX_FLOW_PACKED",
    "PACKED_MIN_SIDE",
    "SCHEMA",
    "FlowTarget",
    "MotionDataConfig",
    "MotionPairProvider",
    "MotionSampleSpec",
    "MotionSampleStream",
    "MotionTrainingConfig",
    "MotionValueCalibration",
    "PlatformDecision",
    "RegionSpec",
    "build_estimator",
    "count_lr_drops",
    "STEM_GROUPS",
    "analytic_stem_delta",
    "decide_platform",
    "flow_target_from_supervision",
    "model_parameter_report",
    "run_stem_report",
    "load_motion_training_config",
    "main",
    "packed_to_sensor",
    "sensor_to_packed",
    "sequence_loss",
]

SCHEMA = "owac.n01_full.p2_motion_train.v1"
CHECKPOINT_SCHEMA = "owac.n01_full.p2_motion_checkpoint.v1"

#: Upstream ``core/loss.py`` provenance (GitHub contents API, fixed commit).
AUTHOR_LOSS_COMMIT = "9137517ba24e628442aec097d3afe71d03503b75"
AUTHOR_LOSS_SHA256 = "12396473603867e32cd1f25ba9123adbcc4709430593df957dd97c24ca41e2fb"
#: Upstream train recipe the LR / iteration tiers are anchored on.
AUTHOR_TRAIN_CONFIG = "config/train/Tartan-C368x496-M.json"
DEFAULT_GAMMA = 0.8
DEFAULT_MAX_FLOW_PACKED = 400.0

#: ``8 * 2 ** corr_levels`` packed cells per side: the vendored 4-level
#: correlation pyramid cannot build a feature map below this at ``iters > 0``.
PACKED_MIN_SIDE = 128

#: Top-level model modules whose gradient must be present after one real step.
GRADIENT_GROUPS = ("cnet", "fnet", "update_block", "flow_head", "init_conv", "upsample_weight")
EXPOSURE_MODES = ("known_metadata_gain", "none")

#: Evidence scale of a run: the authorised formal budget, or a development arm.
RUN_SCALES = ("formal", "dev")
MONITORS = ("val_epe_sensor_px", "val_epe_packed", "val_loss")


# ---------------------------------------------------------------------------
# units, target construction and loss
# ---------------------------------------------------------------------------


def packed_to_sensor(flow_packed: Any, sensor_per_packed: int) -> torch.Tensor:
    """Packed-cell displacement -> sensor-pixel displacement (one packed step = two)."""
    step = _sensor_per_packed(sensor_per_packed)
    return torch.as_tensor(flow_packed, dtype=torch.float32) * float(step)


def sensor_to_packed(flow_sensor_px: Any, sensor_per_packed: int) -> torch.Tensor:
    """Sensor-pixel displacement -> packed-cell displacement (division by the step)."""
    step = _sensor_per_packed(sensor_per_packed)
    return torch.as_tensor(flow_sensor_px, dtype=torch.float32) / float(step)


def _sensor_per_packed(value: int) -> int:
    if isinstance(value, bool) or int(value) not in (1, 2):
        raise ValueError(f"sensor_per_packed must be 1 or 2, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class FlowTarget:
    """Packed-cell pull target of one pair, with its legal valid mask.

    ``flow`` is ``[2, Hp, Wp]`` in packed cells with channel 0 the row (y)
    displacement, matching the model's output order.  ``valid`` is the
    ``[Hp, Wp]`` mask of cells with at least one valid channel;
    ``channel_valid`` keeps the ``[C, Hp, Wp]`` per-channel decision and
    ``channel_flow`` the ``[C, 2, Hp, Wp]`` per-channel packed displacement, so
    the aggregation is auditable and per-phase metrics are possible.
    ``sensor_per_packed`` names the conversion of this stem (2 for cfa4).
    """

    flow: torch.Tensor
    valid: torch.Tensor
    channel_valid: torch.Tensor
    channel_flow: torch.Tensor
    sensor_per_packed: int

    def flow_sensor_px(self) -> torch.Tensor:
        return packed_to_sensor(self.flow, self.sensor_per_packed)


def flow_target_from_supervision(
    supervision: MotionSupervision,
    geometry: FrameGeometry,
    *,
    mode: str,
) -> FlowTarget:
    """Analytic pull truth -> one packed-cell field plus the legal valid mask.

    ``packed_xy_source`` is ``[C, h, w, 2]`` in ``(x, y)`` order on the *source*
    frame's packed grid; the reference cell is the packed index of the matching
    channel, so the displacement is ``target - index`` in packed cells.  Cells
    whose supervision is ``occluded_in_source`` or ``out_of_bounds`` are never
    valid in any channel and are excluded from both the target and the mask.
    """
    if mode not in MODES:
        raise ValueError(f"unknown stem mode {mode!r}; expected {MODES}")
    if mode == "cfa4" and not geometry.is_cfa:
        raise ValueError("mode 'cfa4' needs a CFA geometry")
    if mode == "rgb3" and geometry.is_cfa:
        raise ValueError("mode 'rgb3' needs the three-channel pseudo-RAW geometry")

    packed = supervision.packed_xy_source
    channels = geometry.channels
    rows, cols = geometry.packed_shape
    expected = (channels, rows, cols, 2)
    if tuple(packed.shape) != expected:
        raise ValueError(
            f"packed_xy_source has shape {tuple(packed.shape)}, expected {expected} for this "
            "geometry; the supervision and the frame family disagree"
        )
    sensor_valid = supervision.valid
    if tuple(sensor_valid.shape) != geometry.sensor_shape:
        raise ValueError(
            f"supervision.valid has shape {tuple(sensor_valid.shape)}, expected the sensor grid "
            f"{geometry.sensor_shape}"
        )

    # The reference query of channel k is the packed lattice index (row, col); the
    # source-frame packed target already carries the per-channel sensor offsets, so
    # the displacement is target - index in packed cells of the source frame.
    row_packed = torch.arange(rows, dtype=torch.float32).reshape(rows, 1)
    col_packed = torch.arange(cols, dtype=torch.float32).reshape(1, cols)
    flows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for channel in range(channels):
        row_index, col_index = geometry.grid_indices_yx(channel)
        mask = sensor_valid[row_index, col_index]
        dy = packed[channel, :, :, 1].to(torch.float32) - row_packed
        dx = packed[channel, :, :, 0].to(torch.float32) - col_packed
        flows.append(torch.stack([dy, dx], dim=0))
        masks.append(mask)
    channel_valid = torch.stack(masks, dim=0)
    count = channel_valid.sum(dim=0)
    stacked = torch.stack(flows, dim=0)
    flow = (stacked * channel_valid.unsqueeze(1)).sum(dim=0) / count.clamp_min(1).unsqueeze(0)
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("every cell of this pair is invalid; there is no supervision to fit")
    if not bool(torch.isfinite(flow[..., valid]).all()):
        raise ValueError("the packed target is not finite on its valid mask")
    return FlowTarget(
        flow=flow,
        valid=valid,
        channel_valid=channel_valid,
        channel_flow=stacked,
        sensor_per_packed=int(SENSOR_PER_PACKED[mode]),
    )


def sequence_loss(
    output: Mapping[str, Any],
    flow_gt: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
    max_flow: float = DEFAULT_MAX_FLOW_PACKED,
) -> torch.Tensor:
    """Author ``sequence_loss`` over the iterative predictions.

    ``output['nf'][i]`` is the author mixture-head negative log-likelihood of
    iteration ``i`` computed against ``flow_gt``; the iteration weight is
    ``gamma ** (n_predictions - i - 1)`` and each iteration contributes a masked
    mean over the legal valid mask (``valid`` and ``|flow_gt| < max_flow``).
    ``flow_gt`` is in packed cells, the same unit the model predicts.
    """
    if not math.isfinite(float(gamma)) or not 0.0 < float(gamma) <= 1.0:
        raise ValueError(f"gamma must be finite in (0, 1], got {gamma!r}")
    if not math.isfinite(float(max_flow)) or float(max_flow) <= 0.0:
        raise ValueError(f"max_flow must be finite and positive, got {max_flow!r}")
    if not isinstance(output, Mapping) or "flow" not in output or "nf" not in output:
        raise TypeError("the model output must carry both 'flow' and 'nf' (test_mode=False)")
    nf = output["nf"]
    predictions = output["flow"]
    if nf is None:
        raise ValueError(
            "the model returned no 'nf'; run forward with test_mode=False and a flow_gt so the "
            "author iterative loss exists"
        )
    if len(nf) != len(predictions):
        raise ValueError(
            f"the model returned {len(nf)} nf predictions for {len(predictions)} flow predictions"
        )
    flow_gt = torch.as_tensor(flow_gt)
    if flow_gt.ndim != 4 or flow_gt.shape[1] != 2:
        raise ValueError(
            f"flow_gt must be [N, 2, H, W] in packed cells, got {tuple(flow_gt.shape)}"
        )
    if valid.shape != flow_gt.shape[:1] + flow_gt.shape[2:]:
        raise ValueError(
            f"valid must be [N, H, W] matching flow_gt, got {tuple(valid.shape)} for "
            f"{tuple(flow_gt.shape)}"
        )
    magnitude = torch.sum(flow_gt**2, dim=1).sqrt()
    legal = (valid >= 0.5) & (magnitude < float(max_flow))
    total = flow_gt.new_zeros(())
    for index in range(len(nf)):
        weight = float(gamma) ** (len(nf) - index - 1)
        loss_i = nf[index]
        detached = loss_i.detach()
        final_mask = (~torch.isnan(detached)) & (~torch.isinf(detached)) & legal[:, None]
        total = total + weight * (
            (final_mask * loss_i).sum() / final_mask.sum().to(loss_i.dtype).clamp_min(1.0)
        )
    return total


# ---------------------------------------------------------------------------
# declared value calibration (black/white level, exposure metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionValueCalibration:
    """Declared RAW -> network-range transform for one sensor calibration.

    The interval is the declared black/white level of the noise calibration, not
    a per-image statistic.  With ``exposure_correction='known_metadata_gain'``
    and audited exposure metadata the source frame is placed on the reference
    frame's exposure scale through
    :func:`owac.raw.n01_noise.apply_exposure_gain` (value only; the variance
    bookkeeping of the same gain is the noise module's concern), and both frames
    are relabelled with the derived calibration version.  Unknown exposure stays
    in the raw domain and is reported, never silently corrected.
    """

    black_level_dn: float
    white_level_dn: float
    exposure_correction: str = "known_metadata_gain"
    corrected_suffix: str = "+exposure_normalized"

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.black_level_dn)) and float(self.black_level_dn) >= 0.0):
            raise ValueError("black_level_dn must be finite and non-negative")
        if not (
            math.isfinite(float(self.white_level_dn))
            and float(self.white_level_dn) > float(self.black_level_dn)
        ):
            raise ValueError("white_level_dn must be finite and above the black level")
        if self.exposure_correction not in EXPOSURE_MODES:
            raise ValueError(
                f"unknown exposure_correction {self.exposure_correction!r}; expected "
                f"{EXPOSURE_MODES}"
            )
        if not self.corrected_suffix:
            raise ValueError("corrected_suffix must be a non-empty derived-domain label")

    @classmethod
    def from_data_config(
        cls, config: P2MotionNoiseConfig, *, exposure_correction: str
    ) -> MotionValueCalibration:
        return cls(
            black_level_dn=float(config.noise.black_level_dn),
            white_level_dn=float(config.noise.white_level_dn),
            exposure_correction=exposure_correction,
        )

    def corrected_version(self, raw_version: str) -> str:
        return f"{raw_version}{self.corrected_suffix}"

    def normalization(self, raw_version: str) -> dict[str, RawNormalization]:
        """Normalization specs for the raw domain and (optionally) the corrected one."""
        raw = RawNormalization.affine_calibration(
            self.black_level_dn,
            self.white_level_dn,
            source="p2_motion_noise.json declared black/white level",
            calibration_version=raw_version,
        )
        if self.exposure_correction == "none":
            return {raw_version: raw}
        corrected = RawNormalization.affine_calibration(
            self.black_level_dn,
            self.white_level_dn,
            source=(
                "p2_motion_noise.json declared black/white level; signal scaled to the "
                "reference frame's audited gain"
            ),
            calibration_version=self.corrected_version(raw_version),
        )
        return {raw_version: raw, self.corrected_version(raw_version): corrected}

    def corrected_frames(
        self,
        reference: RawSensorFrame,
        source: RawSensorFrame,
        metadata: Mapping[str, Any],
    ) -> tuple[RawSensorFrame, RawSensorFrame, bool]:
        """Return the pair fed to the network plus whether exposure was audited."""
        exposure = metadata.get("exposure") if isinstance(metadata, Mapping) else None
        exposure = dict(exposure) if isinstance(exposure, Mapping) else {}
        known = bool(exposure.get("known"))
        reference_gain = exposure.get("reference_gain")
        source_gain = exposure.get("source_gain")
        if (
            self.exposure_correction == "none"
            or not known
            or reference_gain is None
            or source_gain is None
        ):
            return reference, source, False
        reference_gain = float(reference_gain)
        source_gain = float(source_gain)
        if not (
            math.isfinite(reference_gain)
            and math.isfinite(source_gain)
            and reference_gain > 0.0
            and source_gain > 0.0
        ):
            raise ValueError(
                "audited exposure gains must be finite and positive, got "
                f"{(reference_gain, source_gain)}"
            )
        ratio = reference_gain / source_gain
        scaled, _ = apply_exposure_gain(
            source.values,
            torch.zeros_like(source.values),
            gain=ratio,
            black_level=self.black_level_dn,
        )
        corrected_source = replace(
            source,
            values=scaled.to(torch.float32),
            calibration_version=self.corrected_version(source.calibration_version),
        )
        corrected_reference = replace(
            reference, calibration_version=self.corrected_version(reference.calibration_version)
        )
        return corrected_reference, corrected_source, True


# ---------------------------------------------------------------------------
# deterministic sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionSampleSpec:
    """One supervised pair request: region, case, sequence token, history gap."""

    split: str
    case: str
    sequence: str
    history_offset: int = 1

    def __post_init__(self) -> None:
        if not self.split:
            raise ValueError("a motion sample needs its region")
        if not self.case or not self.sequence:
            raise ValueError("a motion sample needs its case and sequence token")
        if int(self.history_offset) < 1:
            raise ValueError("history_offset must be at least one frame")

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "case": self.case,
            "sequence": self.sequence,
            "history_offset": int(self.history_offset),
        }

    def identity(self) -> tuple[str, str, str, int]:
        return (self.split, self.case, self.sequence, int(self.history_offset))


class MotionSampleStream:
    """Deterministic permutation + cursor over a fixed list of pair specs.

    The permutation is drawn from a private ``torch.Generator`` that is part of
    the checkpoint, so a resumed run replays exactly the same batch identities
    as a continuous one.
    """

    def __init__(self, specs: Sequence[MotionSampleSpec], *, seed: int) -> None:
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("a sample stream needs at least one spec")
        self._generator = torch.Generator().manual_seed(int(seed) % (1 << 63))
        self._epoch = 0
        self._permutation: list[int] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self.specs)

    def _permute(self) -> None:
        order = torch.randperm(len(self.specs), generator=self._generator)
        self._permutation = [int(value) for value in order.tolist()]
        self._cursor = 0
        self._epoch += 1

    def next(self) -> MotionSampleSpec:
        if self._cursor >= len(self._permutation):
            self._permute()
        spec = self.specs[self._permutation[self._cursor]]
        self._cursor += 1
        return spec

    def state(self) -> dict[str, Any]:
        return {
            "epoch": int(self._epoch),
            "cursor": int(self._cursor),
            "permutation": list(self._permutation),
            "rng_state": self._generator.get_state(),
            "spec_count": len(self.specs),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        if int(state.get("spec_count", -1)) != len(self.specs):
            raise ValueError(
                "the checkpoint sampler covers a different number of specs than this run; "
                "the data configuration changed and the run cannot be continued"
            )
        self._epoch = int(state["epoch"])
        self._cursor = int(state["cursor"])
        permutation = [int(index) for index in state["permutation"]]
        if any(not 0 <= index < len(self.specs) for index in permutation):
            raise ValueError("the checkpoint permutation indexes outside the declared specs")
        self._permutation = permutation
        self._generator.set_state(state["rng_state"])

    def receipt(self) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps([int(index) for index in self._permutation]).encode()
        ).hexdigest()
        return {
            "epoch": int(self._epoch),
            "cursor": int(self._cursor),
            "permutation_sha256": digest,
            "spec_count": len(self.specs),
        }


def _next_permutation_identity(stream: MotionSampleStream) -> tuple[MotionSampleSpec, ...]:
    """Peek the next identities without consuming the stream (test/evidence helper)."""
    saved = stream.state()
    known: list[MotionSampleSpec] = []
    try:
        for _ in range(4):
            known.append(stream.next())
    finally:
        stream.load_state(saved)
    return tuple(known)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _reject_unknown_keys(data: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"{what} has unknown keys {unknown}; allowed keys are {sorted(allowed)}")


def _expand_sequences(payload: Any, *, region: str, case: str) -> tuple[str, ...]:
    if isinstance(payload, Mapping):
        _reject_unknown_keys(payload, ("prefix", "count"), f"{region}.{case} sequences")
        prefix = payload.get("prefix")
        count = payload.get("count")
        if not isinstance(prefix, str) or not prefix or "/" in prefix:
            raise ValueError(f"{region}.{case} needs a non-empty prefix without '/'")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{region}.{case} needs a positive integer count")
        return tuple(f"{prefix}{index}" for index in range(count))
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, Sequence):
        sequences = tuple(str(item) for item in payload)
        if not sequences or any(not item or "/" in item for item in sequences):
            raise ValueError(f"{region}.{case} needs non-empty sequence tokens without '/'")
        return sequences
    raise TypeError(f"{region}.{case} sequences must be a list, a string or a prefix/count object")


@dataclass(frozen=True)
class RegionSpec:
    """One region of the four-way split: which case/sequence pairs it draws."""

    cases: Mapping[str, tuple[str, ...]]
    history_offsets: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("a region needs at least one case")
        for case, sequences in self.cases.items():
            if not isinstance(case, str) or not case:
                raise ValueError("region case names must be non-empty strings")
            if not isinstance(sequences, tuple) or not sequences:
                raise ValueError(f"case {case!r} needs at least one sequence token")
            if case == "translation_subpixel":
                for token in sequences:
                    parse_subpixel_token(token)
        if not self.history_offsets or any(int(offset) < 1 for offset in self.history_offsets):
            raise ValueError("history_offsets must be positive frame gaps")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, region: str) -> RegionSpec:
        _reject_unknown_keys(data, ("cases", "history_offsets"), f"data.{region}")
        cases = data.get("cases")
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError(f"data.{region}.cases must declare at least one case")
        expanded = {
            str(case): _expand_sequences(payload, region=region, case=str(case))
            for case, payload in cases.items()
        }
        offsets = tuple(int(item) for item in data.get("history_offsets", (1,)))
        return cls(cases=expanded, history_offsets=offsets)

    def specs(self, split: str) -> tuple[MotionSampleSpec, ...]:
        return tuple(
            MotionSampleSpec(split=split, case=case, sequence=sequence, history_offset=offset)
            for case, sequences in self.cases.items()
            for sequence in sequences
            for offset in self.history_offsets
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": {case: list(sequences) for case, sequences in self.cases.items()},
            "history_offsets": list(self.history_offsets),
        }


@dataclass(frozen=True)
class MotionDataConfig:
    """Data regions, geometry override and value calibration of a training run.

    The shipped ``p2_motion_noise.json`` geometry (64x64 sensor -> 32x32 packed)
    is below the vendored correlation-pyramid minimum, so this configuration
    declares an explicit geometry override.  ``recipe_scale`` multiplies the
    pixel-valued scene lengths of every case so the declared relative motion and
    occluder size are preserved at the larger grid; the ``translation_subpixel``
    sequence tokens are absolute declarations and are not scaled.
    """

    config_path: str
    input_mode: str | None = None
    geometry: Mapping[str, Any] | None = None
    pseudo_raw_geometry: Mapping[str, Any] | None = None
    recipe_scale: float = 1.0
    exposure_correction: str = "known_metadata_gain"
    batch_size: int = 1
    cache_entries: int = 512
    train: RegionSpec = field(default_factory=RegionSpec)
    calibration: RegionSpec = field(default_factory=RegionSpec)
    validation: RegionSpec = field(default_factory=RegionSpec)
    test: RegionSpec = field(default_factory=RegionSpec)

    def __post_init__(self) -> None:
        if not isinstance(self.config_path, str) or not self.config_path:
            raise ValueError("data.config_path must name the P2 motion/noise JSON")
        if self.input_mode is not None and not isinstance(self.input_mode, str):
            raise TypeError("data.input_mode must be a string or null")
        if not (math.isfinite(float(self.recipe_scale)) and float(self.recipe_scale) > 0.0):
            raise ValueError("data.recipe_scale must be finite and positive")
        if self.exposure_correction not in EXPOSURE_MODES:
            raise ValueError(
                f"data.exposure_correction must be one of {EXPOSURE_MODES}, got "
                f"{self.exposure_correction!r}"
            )
        if isinstance(self.batch_size, bool) or int(self.batch_size) < 1:
            raise ValueError("data.batch_size must be a positive integer")
        if isinstance(self.cache_entries, bool) or int(self.cache_entries) < 0:
            raise ValueError("data.cache_entries must be a non-negative integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionDataConfig:
        allowed = (
            "config_path",
            "input_mode",
            "geometry",
            "pseudo_raw_geometry",
            "recipe_scale",
            "exposure_correction",
            "batch_size",
            "cache_entries",
            "train",
            "calibration",
            "validation",
            "test",
        )
        _reject_unknown_keys(data, allowed, "data")
        for region in ("train", "calibration", "validation", "test"):
            if region not in data:
                raise ValueError(f"data must declare the {region!r} region")
        return cls(
            config_path=str(data["config_path"]),
            input_mode=None if data.get("input_mode") is None else str(data["input_mode"]),
            geometry=dict(data["geometry"]) if data.get("geometry") is not None else None,
            pseudo_raw_geometry=(
                dict(data["pseudo_raw_geometry"])
                if data.get("pseudo_raw_geometry") is not None
                else None
            ),
            recipe_scale=float(data.get("recipe_scale", 1.0)),
            exposure_correction=str(data.get("exposure_correction", "known_metadata_gain")),
            batch_size=int(data.get("batch_size", 1)),
            cache_entries=int(data.get("cache_entries", 512)),
            train=RegionSpec.from_dict(data["train"], region="train"),
            calibration=RegionSpec.from_dict(data["calibration"], region="calibration"),
            validation=RegionSpec.from_dict(data["validation"], region="validation"),
            test=RegionSpec.from_dict(data["test"], region="test"),
        )

    def region(self, split: str) -> RegionSpec:
        try:
            return {
                "train": self.train,
                "calibration": self.calibration,
                "validation": self.validation,
                "test": self.test,
            }[split]
        except KeyError as error:
            raise KeyError(f"unknown region {split!r}") from error

    def resolved_data_config(self) -> P2MotionNoiseConfig:
        """Load the P2 motion/noise JSON, apply the geometry override and recipe scale."""
        base = load_p2_motion_noise_config(self.config_path)
        overrides: dict[str, Any] = {}
        if self.geometry is not None:
            overrides["geometry"] = FrameGeometry.from_dict(self.geometry)
        if self.pseudo_raw_geometry is not None:
            overrides["pseudo_raw_geometry"] = FrameGeometry.from_dict(self.pseudo_raw_geometry)
        if float(self.recipe_scale) != 1.0:
            overrides["cases"] = {
                case: _scaled_recipe(recipe, float(self.recipe_scale))
                for case, recipe in base.cases.items()
            }
        if not overrides:
            return base
        return replace(base, **overrides)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "input_mode": self.input_mode,
            "geometry": dict(self.geometry) if self.geometry is not None else None,
            "pseudo_raw_geometry": (
                dict(self.pseudo_raw_geometry) if self.pseudo_raw_geometry is not None else None
            ),
            "recipe_scale": float(self.recipe_scale),
            "exposure_correction": self.exposure_correction,
            "batch_size": int(self.batch_size),
            "cache_entries": int(self.cache_entries),
            "train": self.train.as_dict(),
            "calibration": self.calibration.as_dict(),
            "validation": self.validation.as_dict(),
            "test": self.test.as_dict(),
        }


#: Pixel-valued scene lengths scaled together by ``data.recipe_scale``.
_SCALED_RECIPE_FIELDS = (
    "occluder_radius",
    "warp_amplitude_px",
    "warp_sigma_px",
    "blur_length_px",
    "texture_period_px",
    "jitter_px",
)


def _scaled_recipe(recipe: SceneRecipe, scale: float) -> SceneRecipe:
    scaled = {name: float(getattr(recipe, name)) * scale for name in _SCALED_RECIPE_FIELDS}
    scaled["background_step_px"] = tuple(
        float(value) * scale for value in recipe.background_step_px
    )
    scaled["object_step_px"] = tuple(float(value) * scale for value in recipe.object_step_px)
    return replace(recipe, **scaled)


@dataclass(frozen=True)
class MotionModelConfig:
    """Stem, width and iterative-update count of the motion network."""

    mode: str = "cfa4"
    depth: str = "resnet34"
    iters: int = 4
    pretrained_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"model.mode must be one of {MODES}, got {self.mode!r}")
        if self.depth not in ("resnet18", "resnet34"):
            raise ValueError(f"model.depth must be resnet18 or resnet34, got {self.depth!r}")
        if isinstance(self.iters, bool) or int(self.iters) < 0:
            raise ValueError("model.iters must be a non-negative integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionModelConfig:
        _reject_unknown_keys(data, ("mode", "depth", "iters", "pretrained_checkpoint"), "model")
        checkpoint = data.get("pretrained_checkpoint")
        return cls(
            mode=str(data.get("mode", "cfa4")),
            depth=str(data.get("depth", "resnet34")),
            iters=int(data.get("iters", 4)),
            pretrained_checkpoint=None if checkpoint is None else str(checkpoint),
        )


@dataclass(frozen=True)
class MotionSelectionConfig:
    """Two LR tiers x the author's reasonable iteration tiers, same budget."""

    lrs: tuple[float, ...] = (2.5e-4, 5.0e-5)
    iters: tuple[int, ...] = (4, 8)
    updates: int = 256

    def __post_init__(self) -> None:
        if len(self.lrs) != 2:
            raise ValueError("the development selection declares exactly two LR tiers")
        if len(set(self.lrs)) != len(self.lrs) or any(
            not math.isfinite(float(lr)) or float(lr) <= 0.0 for lr in self.lrs
        ):
            raise ValueError("selection lrs must be distinct finite positive numbers")
        if not self.iters or len(set(self.iters)) != len(self.iters):
            raise ValueError("selection iters must be distinct")
        if any(int(value) < 0 for value in self.iters):
            raise ValueError("selection iters must be non-negative")
        if isinstance(self.updates, bool) or int(self.updates) < 1:
            raise ValueError("selection.updates must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionSelectionConfig:
        _reject_unknown_keys(data, ("lrs", "iters", "updates"), "selection")
        return cls(
            lrs=tuple(float(item) for item in data.get("lrs", (2.5e-4, 5.0e-5))),
            iters=tuple(int(item) for item in data.get("iters", (4, 8))),
            updates=int(data.get("updates", 256)),
        )


@dataclass(frozen=True)
class MotionBudgetConfig:
    """Formal budget, validation cadence and the locked plateau threshold."""

    updates: int = 100_000
    max_updates: int = 300_000
    validation_interval: int = 1_000
    plateau_checks: int = 8
    delta_val: float = 0.002
    monitor: str = "val_epe_sensor_px"

    def __post_init__(self) -> None:
        if isinstance(self.updates, bool) or int(self.updates) < 1:
            raise ValueError("budget.updates must be positive")
        if int(self.max_updates) < int(self.updates):
            raise ValueError("budget.max_updates must not be below budget.updates")
        if int(self.validation_interval) < 1:
            raise ValueError("budget.validation_interval must be positive")
        if int(self.plateau_checks) != 8:
            raise ValueError(
                "validation_plateau is declared after 8 consecutive improvements below "
                f"delta_val; got plateau_checks={self.plateau_checks}"
            )
        if not math.isfinite(float(self.delta_val)) or float(self.delta_val) <= 0.0:
            raise ValueError("budget.delta_val must be finite and positive")
        if self.monitor not in MONITORS:
            raise ValueError(f"budget.monitor must be one of {MONITORS}, got {self.monitor!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionBudgetConfig:
        _reject_unknown_keys(
            data,
            (
                "updates",
                "max_updates",
                "validation_interval",
                "plateau_checks",
                "delta_val",
                "monitor",
            ),
            "budget",
        )
        return cls(**{key: data[key] for key in data})


@dataclass(frozen=True)
class MotionScheduleConfig:
    """Piecewise-constant LR schedule with at least two effective drops.

    Drops are declared at **absolute update numbers**, not as fractions of the
    budget: extending a run must never move an already-executed LR trajectory,
    so a resumed segment replays the same optimiser path it would have followed
    continuously.  Each drop multiplies the running factor, and at least two
    drops are required by the platform contract.
    """

    drops: tuple[tuple[int, float], ...] = ((50_000, 0.1), (80_000, 0.01))
    min_drop_ratio: float = 0.1

    def __post_init__(self) -> None:
        if len(self.drops) < 2:
            raise ValueError(
                "the platform criterion needs at least two effective LR drops; declare at "
                "least two drops"
            )
        previous_update = -1
        for at, factor in self.drops:
            if isinstance(at, bool) or int(at) < 1 or int(at) <= previous_update:
                raise ValueError(
                    f"schedule drop updates must be strictly increasing positive integers, got "
                    f"{self.drops!r}"
                )
            if not math.isfinite(float(factor)) or not 0.0 < float(factor) < 1.0:
                raise ValueError(f"every schedule factor must be in (0, 1), got {factor!r}")
            previous_update = int(at)
        if not 0.0 < float(self.min_drop_ratio) < 1.0:
            raise ValueError("min_drop_ratio must be in (0, 1)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionScheduleConfig:
        _reject_unknown_keys(data, ("drops", "min_drop_ratio"), "schedule")
        default = ((50_000, 0.1), (80_000, 0.01))
        drops = tuple((int(at), float(factor)) for at, factor in data.get("drops", default))
        return cls(drops=drops, min_drop_ratio=float(data.get("min_drop_ratio", 0.1)))

    def phase_index(self, update: int) -> int:
        """Index of the active (completed-drop) phase; 0 is the initial LR."""
        index = 0
        for position, (at, _factor) in enumerate(self.drops):
            if int(update) >= int(at):
                index = position + 1
        return index

    def lr_at(self, update: int, base_lr: float) -> float:
        factor = 1.0
        for at, drop_factor in self.drops:
            if int(update) >= int(at):
                factor *= float(drop_factor)
        return float(base_lr) * factor

    def is_final_phase(self, update: int) -> bool:
        return int(update) >= int(self.drops[-1][0])

    def as_list(self) -> list[list[float]]:
        return [[float(at), float(factor)] for at, factor in self.drops]


def count_lr_drops(lrs: Sequence[float], *, min_drop_ratio: float = 0.1) -> int:
    """Effective LR reductions: falls of at least ``min_drop_ratio`` below the last level."""
    if not 0.0 < float(min_drop_ratio) < 1.0:
        raise ValueError("min_drop_ratio must be in (0, 1)")
    drops = 0
    reference = None
    for value in lrs:
        lr = float(value)
        if reference is None or lr > reference:
            reference = lr
        elif lr < reference * (1.0 - float(min_drop_ratio)):
            drops += 1
            reference = lr
    return drops


@dataclass(frozen=True)
class MotionOptimizerConfig:
    """AdamW hyper-parameters, gradient clip and the author loss knobs."""

    weight_decay: float = 1e-4
    eps: float = 1e-8
    clip_grad: float = 1.0
    gamma: float = DEFAULT_GAMMA
    max_flow_packed: float = DEFAULT_MAX_FLOW_PACKED
    amp: str = "bf16"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0.0:
            raise ValueError("optimizer.weight_decay must be finite and non-negative")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("optimizer.eps must be finite and positive")
        if not math.isfinite(float(self.clip_grad)) or float(self.clip_grad) <= 0.0:
            raise ValueError("optimizer.clip_grad must be finite and positive")
        if self.amp not in ("off", "bf16", "fp16"):
            raise ValueError(f"optimizer.amp must be off, bf16 or fp16, got {self.amp!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionOptimizerConfig:
        _reject_unknown_keys(
            data,
            ("weight_decay", "eps", "clip_grad", "gamma", "max_flow_packed", "amp"),
            "optimizer",
        )
        return cls(**{key: data[key] for key in data})


@dataclass(frozen=True)
class MotionCheckpointConfig:
    """Checkpoint cadence; ``best`` always follows the validation monitor."""

    every: int = 1_000
    keep_last: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or int(self.every) < 1:
            raise ValueError("checkpoint.every must be positive")
        if isinstance(self.keep_last, bool) or int(self.keep_last) < 1:
            raise ValueError("checkpoint.keep_last must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionCheckpointConfig:
        _reject_unknown_keys(data, ("every", "keep_last"), "checkpoint")
        return cls(**{key: data[key] for key in data})


@dataclass(frozen=True)
class MotionDiagnosticsConfig:
    """Small-set fit / input / gradient diagnostics that gate ``validation_plateau``."""

    fit_updates: int = 64
    check_updates: int = 256
    fit_batch: int = 2
    fit_max_ratio: float = 0.5
    fit_target: float = 12.0
    input_limit: float = 8.0

    def __post_init__(self) -> None:
        for name in ("fit_updates", "check_updates", "fit_batch"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise ValueError(f"diagnostics.{name} must be positive")
        if not 0.0 < float(self.fit_max_ratio) < 1.0:
            raise ValueError("diagnostics.fit_max_ratio must be in (0, 1)")
        if not math.isfinite(float(self.fit_target)):
            raise ValueError("diagnostics.fit_target must be finite")
        if not math.isfinite(float(self.input_limit)) or float(self.input_limit) <= 0.0:
            raise ValueError("diagnostics.input_limit must be finite and positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionDiagnosticsConfig:
        _reject_unknown_keys(
            data,
            (
                "fit_updates",
                "check_updates",
                "fit_batch",
                "fit_max_ratio",
                "fit_target",
                "input_limit",
            ),
            "diagnostics",
        )
        return cls(**{key: data[key] for key in data})


@dataclass(frozen=True)
class MotionEvaluationConfig:
    """Evaluation knobs shared by ``owac.evaluation.n01_motion``."""

    buckets: Mapping[str, str] = field(default_factory=dict)
    simplified_search_px: int = 8
    risk_bins: int = 4
    risk_quantile: float = 0.9
    risk_margin_sensor_px: float = 1.0
    risk_tolerance_sensor_px: float = 1.0
    static_false_rejection_sensor_px: float = 0.5
    latency_warmup: int = 2
    latency_repeats: int = 5
    max_calibration_pairs: int = 32
    max_validation_pairs: int = 32
    max_test_pairs: int = 32
    display_batch: int = 4

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("evaluation.buckets must map every declared case to a bucket")
        if not 0.0 < float(self.risk_quantile) < 1.0:
            raise ValueError("evaluation.risk_quantile must be in (0, 1)")
        if int(self.risk_bins) < 1:
            raise ValueError("evaluation.risk_bins must be positive")
        if int(self.simplified_search_px) < 0:
            raise ValueError("evaluation.simplified_search_px must be non-negative")
        for name in (
            "latency_warmup",
            "latency_repeats",
            "max_calibration_pairs",
            "max_validation_pairs",
            "max_test_pairs",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"evaluation.{name} must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionEvaluationConfig:
        _reject_unknown_keys(
            data,
            (
                "buckets",
                "simplified_search_px",
                "risk_bins",
                "risk_quantile",
                "risk_margin_sensor_px",
                "risk_tolerance_sensor_px",
                "static_false_rejection_sensor_px",
                "latency_warmup",
                "latency_repeats",
                "max_calibration_pairs",
                "max_validation_pairs",
                "max_test_pairs",
                "display_batch",
            ),
            "evaluation",
        )
        payload = dict(data)
        payload["buckets"] = {
            str(key): str(value) for key, value in data.get("buckets", {}).items()
        }
        return cls(**payload)

    def bucket_of(self, case: str) -> str:
        try:
            return self.buckets[case]
        except KeyError as error:
            raise KeyError(f"case {case!r} has no declared evaluation bucket") from error


@dataclass(frozen=True)
class MotionTrainingConfig:
    """The complete P2 motion training configuration (strict JSON, no unknown keys).

    ``scale`` labels the run's evidence scale: ``"formal"`` for the authorised
    100k budget and ``"dev"`` for a development-scale arm (for example the
    three-channel pseudo-RAW stem).  It is a reporting label, not a run-defining
    section: it stays out of ``core_fingerprint`` so labelling a config cannot
    invalidate a checkpoint.
    """

    schema: str
    stage: str
    scale: str
    seeds: tuple[int, ...]
    model: MotionModelConfig
    data: MotionDataConfig
    selection: MotionSelectionConfig
    budget: MotionBudgetConfig
    schedule: MotionScheduleConfig
    optimizer: MotionOptimizerConfig
    checkpoint: MotionCheckpointConfig
    diagnostics: MotionDiagnosticsConfig
    evaluation: MotionEvaluationConfig
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unknown schema {self.schema!r}; expected {SCHEMA!r}")
        if not self.stage:
            raise ValueError("stage must be declared")
        if self.scale not in RUN_SCALES:
            raise ValueError(f"scale must be one of {RUN_SCALES}, got {self.scale!r}")
        if len(self.seeds) < 1 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and non-empty")
        if any(int(seed) < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(self.evaluation.buckets.values())) < 4:
            raise ValueError(
                "the evaluation must keep all four declared buckets (fast-alignable, fast-hard, "
                "slow-structural-change, current-frame-anomaly)"
            )
        if int(self.schedule.drops[-1][0]) >= int(self.budget.updates):
            raise ValueError(
                f"the final LR drop at update {self.schedule.drops[-1][0]} must fall inside the "
                f"planned budget of {self.budget.updates} updates, otherwise a plateau could "
                "never be reached"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotionTrainingConfig:
        _reject_unknown_keys(
            data,
            (
                "schema",
                "stage",
                "scale",
                "seeds",
                "model",
                "data",
                "selection",
                "budget",
                "schedule",
                "optimizer",
                "checkpoint",
                "diagnostics",
                "evaluation",
                "notes",
            ),
            "motion training config",
        )
        for required in (
            "schema",
            "stage",
            "seeds",
            "model",
            "data",
            "selection",
            "budget",
            "schedule",
            "optimizer",
            "checkpoint",
            "diagnostics",
            "evaluation",
        ):
            if required not in data:
                raise ValueError(
                    f"motion training config is missing the required field {required!r}"
                )
        return cls(
            schema=str(data["schema"]),
            stage=str(data["stage"]),
            scale=str(data.get("scale", "formal")),
            seeds=tuple(int(seed) for seed in data["seeds"]),
            model=MotionModelConfig.from_dict(data["model"]),
            data=MotionDataConfig.from_dict(data["data"]),
            selection=MotionSelectionConfig.from_dict(data["selection"]),
            budget=MotionBudgetConfig.from_dict(data["budget"]),
            schedule=MotionScheduleConfig.from_dict(data["schedule"]),
            optimizer=MotionOptimizerConfig.from_dict(data["optimizer"]),
            checkpoint=MotionCheckpointConfig.from_dict(data["checkpoint"]),
            diagnostics=MotionDiagnosticsConfig.from_dict(data["diagnostics"]),
            evaluation=MotionEvaluationConfig.from_dict(data["evaluation"]),
            notes=str(data.get("notes", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage": self.stage,
            "scale": self.scale,
            "seeds": list(self.seeds),
            "model": {
                "mode": self.model.mode,
                "depth": self.model.depth,
                "iters": int(self.model.iters),
                "pretrained_checkpoint": self.model.pretrained_checkpoint,
            },
            "data": self.data.as_dict(),
            "selection": {
                "lrs": list(self.selection.lrs),
                "iters": list(self.selection.iters),
                "updates": int(self.selection.updates),
            },
            "budget": {
                "updates": int(self.budget.updates),
                "max_updates": int(self.budget.max_updates),
                "validation_interval": int(self.budget.validation_interval),
                "plateau_checks": int(self.budget.plateau_checks),
                "delta_val": float(self.budget.delta_val),
                "monitor": self.budget.monitor,
            },
            "schedule": {
                "drops": self.schedule.as_list(),
                "min_drop_ratio": float(self.schedule.min_drop_ratio),
            },
            "optimizer": {
                "weight_decay": float(self.optimizer.weight_decay),
                "eps": float(self.optimizer.eps),
                "clip_grad": float(self.optimizer.clip_grad),
                "gamma": float(self.optimizer.gamma),
                "max_flow_packed": float(self.optimizer.max_flow_packed),
                "amp": self.optimizer.amp,
            },
            "checkpoint": {
                "every": int(self.checkpoint.every),
                "keep_last": int(self.checkpoint.keep_last),
            },
            "diagnostics": {
                "fit_updates": int(self.diagnostics.fit_updates),
                "check_updates": int(self.diagnostics.check_updates),
                "fit_batch": int(self.diagnostics.fit_batch),
                "fit_max_ratio": float(self.diagnostics.fit_max_ratio),
                "fit_target": float(self.diagnostics.fit_target),
                "input_limit": float(self.diagnostics.input_limit),
            },
            "evaluation": {
                "buckets": dict(self.evaluation.buckets),
                "simplified_search_px": int(self.evaluation.simplified_search_px),
                "risk_bins": int(self.evaluation.risk_bins),
                "risk_quantile": float(self.evaluation.risk_quantile),
                "risk_margin_sensor_px": float(self.evaluation.risk_margin_sensor_px),
                "risk_tolerance_sensor_px": float(self.evaluation.risk_tolerance_sensor_px),
                "static_false_rejection_sensor_px": float(
                    self.evaluation.static_false_rejection_sensor_px
                ),
                "latency_warmup": int(self.evaluation.latency_warmup),
                "latency_repeats": int(self.evaluation.latency_repeats),
                "max_calibration_pairs": int(self.evaluation.max_calibration_pairs),
                "max_validation_pairs": int(self.evaluation.max_validation_pairs),
                "max_test_pairs": int(self.evaluation.max_test_pairs),
                "display_batch": int(self.evaluation.display_batch),
            },
            "notes": self.notes,
        }

    @property
    def core_fingerprint(self) -> str:
        """Identity of the run-defining sections; a resume must match it exactly."""
        payload = {
            "model": self.as_dict()["model"],
            "data": self.as_dict()["data"],
            "schedule": self.as_dict()["schedule"],
            "optimizer": self.as_dict()["optimizer"],
            "selection": self.as_dict()["selection"],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_motion_training_config(path: str | Path) -> MotionTrainingConfig:
    """Load and validate ``configs/encoders/n01_full/p2_motion_train.json``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return MotionTrainingConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_hashes(config_path: str | Path | None = None) -> dict[str, str]:
    """Hashes of the source files this run depends on, plus the config when given."""
    root = Path(__file__).resolve().parents[3]
    files = {
        "training/n01_motion.py": Path(__file__).resolve(),
        "evaluation/n01_motion.py": root / "src/owac/evaluation/n01_motion.py",
        "backends/raw_correspondence.py": root / "src/owac/backends/raw_correspondence.py",
        "raw/n01_motion_data.py": root / "src/owac/raw/n01_motion_data.py",
        "raw/n01_noise.py": root / "src/owac/raw/n01_noise.py",
        "frameworks/n01_full/statistics.py": root / "src/owac/frameworks/n01_full/statistics.py",
        "vendor/sea_raft/raft.py": root / "src/owac/baselines/_vendor/sea_raft/raft.py",
        "vendor/sea_raft/provenance.json": root
        / "src/owac/baselines/_vendor/sea_raft/provenance.json",
    }
    hashes = {name: _sha256_file(path) for name, path in files.items() if path.is_file()}
    if config_path is not None:
        hashes["config"] = _sha256_file(config_path)
    return hashes


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# pair provider: render + declared value transform + target construction
# ---------------------------------------------------------------------------


class MotionPairProvider:
    """Deterministic supervised pairs rendered through the declared value path.

    The network inputs come from :meth:`CorrespondenceEstimator.preprocess`, so
    training and evaluation share exactly one value transform.  Rendered pairs
    are cached as CPU tensors (rendering dominates the cost of a step); the
    cache is keyed by the pair identity, so it cannot change a result.
    """

    def __init__(
        self,
        generator: MotionDataGenerator,
        calibration: MotionValueCalibration,
        estimator: CorrespondenceEstimator,
        *,
        input_mode: str,
        mode: str,
        cache_entries: int = 0,
    ) -> None:
        self.generator = generator
        self.calibration = calibration
        self.estimator = estimator
        self.input_mode = input_mode
        self.mode = mode
        self.geometry = generator.geometry(input_mode)
        self.cache_entries = int(cache_entries)
        self._cache: dict[tuple[str, str, str, int, str], tuple[torch.Tensor, ...]] = {}
        self.stats = {
            "rendered": 0,
            "cache_hits": 0,
            "exposure_known": 0,
            "exposure_unknown": 0,
            "pairs": 0,
        }
        self._check_geometry_flag = True

    def tensors(
        self, spec: MotionSampleSpec
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (*spec.identity(), self.input_mode)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        pair = self.generator.sample_pair(
            split=spec.split,
            case=spec.case,
            sequence=spec.sequence,
            history_offset=int(spec.history_offset),
            input_mode=self.input_mode,
        )
        self._check_frames(pair.visible.reference, pair.visible.source)
        reference, source, exposure_known = self.calibration.corrected_frames(
            pair.visible.reference, pair.visible.source, pair.visible.metadata
        )
        reference_input = self.estimator.preprocess(reference).to("cpu")
        source_input = self.estimator.preprocess(source).to("cpu")
        target = flow_target_from_supervision(pair.supervision, self.geometry, mode=self.mode)
        if not bool(torch.isfinite(target.flow).all()):
            raise ValueError(f"pair {spec.identity()} produced a non-finite packed target")
        captured = (reference_input, source_input, target.flow.to(torch.float32), target.valid)
        self.stats["rendered"] += 1
        self.stats["pairs"] += 1
        self.stats["exposure_known" if exposure_known else "exposure_unknown"] += 1
        if self.cache_entries:
            if len(self._cache) >= self.cache_entries:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = captured
        return captured

    def _check_frames(self, reference: RawSensorFrame, source: RawSensorFrame) -> None:
        if not self._check_geometry_flag:
            return
        geometry = self.geometry
        for frame in (reference, source):
            if tuple(frame.packed_shape) != geometry.packed_shape:
                raise ValueError(
                    f"frame packed grid {tuple(frame.packed_shape)} does not match the declared "
                    f"geometry {geometry.packed_shape}"
                )
            if tuple(frame.channel_phase) != tuple(geometry.channel_phase):
                raise ValueError("frame channel_phase does not match the declared geometry")
            if tuple(frame.sensor_origin_xy) != tuple(geometry.origin_yx):
                raise ValueError("frame sensor origin does not match the declared geometry")
        self._check_geometry_flag = False

    def receipt(self) -> dict[str, Any]:
        return {
            "pairs_rendered": self.stats["rendered"],
            "cache_hits": self.stats["cache_hits"],
            "exposure_known_pairs": self.stats["exposure_known"],
            "exposure_unknown_pairs": self.stats["exposure_unknown"],
            "packed_shape": list(self.geometry.packed_shape),
            "sensor_shape": list(self.geometry.sensor_shape),
            "sensor_per_packed": int(SENSOR_PER_PACKED[self.mode]),
        }


# ---------------------------------------------------------------------------
# platform decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformDecision:
    """Machine-readable platform verdict with the conditions that decided it."""

    status: str
    reasons: tuple[str, ...]
    details: Mapping[str, Any]


def decide_platform(
    *,
    val_history: Sequence[Mapping[str, Any]],
    budget: int,
    interval: int,
    monitor: str,
    delta_val: float,
    plateau_checks: int,
    lr_drops: int,
    final_phase: bool,
    diagnostics_passed: bool,
) -> PlatformDecision:
    """Apply the shared-contract platform criteria to a validation history.

    ``improvement`` at a validation point is how much better it is than the best
    value seen before it (zero when it is not a new best).

    A ``validation_plateau`` needs **all** of the following, and every unmet
    condition is named individually in ``reasons``:

    * ``real_improvement_over_run`` -- ``first_value - best_value`` is strictly
      greater than ``delta_val``: the run really improved at some point.  A
      constant curve, or one that only degrades from the first validation, can
      never be a plateau (both have an improvement of zero).
    * ``tail_within_delta_val_of_best`` -- ``|final_value - best_value|`` is at
      most ``delta_val`` (the same locked threshold, no new parameter): a run
      that improved and then drifted away from its best is not "at a plateau".
    * ``flat_tail_below_delta_val`` -- the last ``plateau_checks`` improvements
      are each below ``delta_val``.  This stays literal: slow but persistent
      improvement below ``delta_val`` still counts, the tail does not have to be
      exactly flat.
    * ``two_effective_lr_drops`` and ``final_low_lr_phase`` -- at least two
      effective LR drops were applied and the run is in the final low-LR phase.
    * ``best_not_at_budget_end`` -- the best checkpoint is not inside the last
      ``plateau_checks`` validation windows of the budget.
    * ``diagnostics_passed`` -- the small-set fit / input / gradient diagnostics
      passed.

    Anything else is ``budget_exhausted``; the status is a stop condition, not a
    research verdict.
    """
    if monitor not in MONITORS:
        raise ValueError(f"unknown monitor {monitor!r}; expected {MONITORS}")
    if interval < 1:
        raise ValueError("interval must be positive")
    values = [float(point[monitor]) for point in val_history]
    if len(values) < 2:
        return PlatformDecision(
            status="budget_exhausted",
            reasons=(f"fewer than two validation points ({len(values)}); no plateau is claimable",),
            details={"validations": len(values)},
        )
    running_best = values[0]
    improvements = [float("inf")]
    for value in values[1:]:
        improvements.append(max(0.0, running_best - value))
        running_best = min(running_best, value)
    best_index = min(range(len(values)), key=lambda index: (values[index], index))
    best_value = values[best_index]
    best_update = int(val_history[best_index]["update"])
    tail = improvements[1:]
    flat_tail = len(tail) >= plateau_checks and all(
        improvement < float(delta_val) for improvement in tail[-plateau_checks:]
    )
    best_not_at_end = (int(budget) - best_update) >= plateau_checks * int(interval)
    # D1: an improvement of exactly zero (constant curve, or one that only gets
    # worse) is not an improvement, and a tail that moved away from its best is
    # drift, not a plateau.  Both reuse the locked delta_val.
    run_improvement = values[0] - best_value
    tail_drift = abs(values[-1] - best_value)
    checks = {
        "real_improvement_over_run": run_improvement > float(delta_val),
        "tail_within_delta_val_of_best": tail_drift <= float(delta_val),
        "two_effective_lr_drops": int(lr_drops) >= 2,
        "final_low_lr_phase": bool(final_phase),
        "flat_tail_below_delta_val": flat_tail,
        "best_not_at_budget_end": best_not_at_end,
        "diagnostics_passed": bool(diagnostics_passed),
    }
    details = {
        "validations": len(values),
        "first_value": values[0],
        "final_value": values[-1],
        "best_value": best_value,
        "best_update": best_update,
        "run_improvement": run_improvement,
        "tail_drift": tail_drift,
        "last_improvements": [float(item) for item in tail[-plateau_checks:]],
        "lr_drops": int(lr_drops),
        "conditions": checks,
        "condition_descriptions": {
            "real_improvement_over_run": (
                "first_value - best_value must be strictly greater than delta_val"
            ),
            "tail_within_delta_val_of_best": (
                "|final_value - best_value| must not exceed delta_val"
            ),
            "two_effective_lr_drops": "at least two effective LR drops must have been applied",
            "final_low_lr_phase": "the run must be in the final low-LR phase",
            "flat_tail_below_delta_val": (
                "the last plateau_checks improvements must each be below delta_val"
            ),
            "best_not_at_budget_end": (
                "the best checkpoint must not sit inside the last plateau_checks validation "
                "windows of the budget"
            ),
            "diagnostics_passed": "the small-set fit / input / gradient diagnostics must pass",
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if not failed:
        return PlatformDecision(
            status="validation_plateau",
            reasons=("all platform conditions satisfied",),
            details=details,
        )
    return PlatformDecision(
        status="budget_exhausted",
        reasons=tuple(f"platform condition failed: {name}" for name in failed),
        details=details,
    )


# ---------------------------------------------------------------------------
# trainer session
# ---------------------------------------------------------------------------


def build_estimator(
    config: MotionTrainingConfig,
    data_config: P2MotionNoiseConfig,
    *,
    device: str | torch.device,
    iters: int | None = None,
    normalization: Mapping[str, RawNormalization] | None = None,
) -> CorrespondenceEstimator:
    """Construct the RAW adapter on the resolved data geometry and stem."""
    geometry = data_config.geometry_for(config.data.input_mode or data_config.geometry.input_mode)
    mode = "cfa4" if geometry.is_cfa else "rgb3"
    if mode != config.model.mode:
        raise ValueError(
            f"the data geometry is a {mode} family but model.mode is {config.model.mode!r}"
        )
    return CorrespondenceEstimator(
        mode=mode,
        depth=config.model.depth,
        iters=config.model.iters if iters is None else int(iters),
        device=str(device),
        normalization=normalization,
        pretrained_checkpoint=config.model.pretrained_checkpoint,
    )


class MotionTrainer:
    """One reproducible training session: model, optimizer, stream and receipts."""

    def __init__(
        self,
        config: MotionTrainingConfig,
        *,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        lr: float | None = None,
        iters: int | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.seed = int(config.seeds[0] if seed is None else seed)
        self.base_lr = float(config.selection.lrs[0] if lr is None else lr)
        if not math.isfinite(self.base_lr) or self.base_lr <= 0.0:
            raise ValueError(f"the base learning rate must be positive, got {self.base_lr!r}")
        self.data_config = config.data.resolved_data_config()
        self.input_mode = config.data.input_mode or self.data_config.geometry.input_mode
        self.geometry = self.data_config.geometry_for(self.input_mode)
        self.mode = "cfa4" if self.geometry.is_cfa else "rgb3"
        self.sensor_per_packed = int(SENSOR_PER_PACKED[self.mode])
        self.iters = int(config.model.iters if iters is None else iters)
        if self.iters < 0:
            raise ValueError("iters must be non-negative")
        rows, cols = self.geometry.packed_shape
        if rows % 8 or cols % 8:
            raise ValueError(
                f"the vendored model pads to multiples of eight; the packed grid "
                f"{(rows, cols)} does not satisfy that at no-padding shape"
            )
        if self.iters > 0 and min(rows, cols) < PACKED_MIN_SIDE:
            raise ValueError(
                f"the {min(rows, cols)}-cell packed grid is below the vendored correlation "
                f"pyramid minimum {PACKED_MIN_SIDE} at iters>0; declare a larger data geometry "
                "or iters=0"
            )
        self._seed_everything()
        self.calibration = MotionValueCalibration.from_data_config(
            self.data_config, exposure_correction=config.data.exposure_correction
        )
        self.normalization = self.calibration.normalization(
            self.data_config.noise.calibration_version
        )
        self.estimator = build_estimator(
            config,
            self.data_config,
            device=self.device,
            iters=self.iters,
            normalization=self.normalization,
        )
        self.model = self.estimator.model
        self.params = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not self.params:
            raise RuntimeError("the motion model declares no trainable parameter")
        self.optimizer = torch.optim.AdamW(
            self.params,
            lr=self.base_lr,
            weight_decay=float(config.optimizer.weight_decay),
            eps=float(config.optimizer.eps),
        )
        self._amp = "off" if self.device.type != "cuda" else config.optimizer.amp
        self._amp_dtype = torch.bfloat16 if self._amp == "bf16" else torch.float16
        self._use_amp = self._amp != "off"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp == "fp16")
        self.train_specs = config.data.train.specs("train")
        self.validation_specs = config.data.validation.specs("validation")
        self.calibration_specs = config.data.calibration.specs("calibration")
        self.test_specs = config.data.test.specs("test")
        self.stream = MotionSampleStream(self.train_specs, seed=self.seed + 1)
        self.provider = MotionPairProvider(
            MotionDataGenerator(config=self.data_config),
            self.calibration,
            self.estimator,
            input_mode=self.input_mode,
            mode=self.mode,
            cache_entries=self.config.data.cache_entries,
        )
        self.lr_history: list[float] = []
        self.train_losses: list[float] = []
        self.batch_log: list[list[dict[str, Any]]] = []
        self.val_history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.consumed_samples = 0
        self.diagnostic_samples = 0
        self.lr_drops = 0
        self._lr_reference: float | None = None
        self._param_before = self._parameter_snapshot()
        self.grad_evidence: dict[str, Any] = {}
        self.diagnostics: dict[str, Any] | None = None
        self.start_update = 0
        self.wall_seconds = 0.0

    # -- seeds and parameter evidence -----------------------------------------------------

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed % (1 << 32))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _parameter_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.model.state_dict().items()}

    def parameter_update_evidence(self) -> dict[str, Any]:
        total = 0.0
        per_group: dict[str, float] = {}
        for name, tensor in self.model.state_dict().items():
            before = self._param_before.get(name)
            if before is None or before.shape != tensor.shape:
                continue
            delta = float((tensor.detach().cpu() - before.cpu()).abs().sum())
            total += delta
            group = name.split(".")[0]
            per_group[group] = per_group.get(group, 0.0) + delta
        return {
            "total_abs_update": total,
            "per_group": per_group,
            "updates": self.consumed_samples // max(1, int(self.config.data.batch_size)),
        }

    # -- single update --------------------------------------------------------------------

    def _batch_tensors(
        self, specs: Sequence[MotionSampleSpec]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        entries = [self.provider.tensors(spec) for spec in specs]
        reference = torch.cat([entry[0] for entry in entries], dim=0).to(self.device)
        source = torch.cat([entry[1] for entry in entries], dim=0).to(self.device)
        flow = torch.stack([entry[2] for entry in entries], dim=0).to(self.device)
        valid = torch.stack([entry[3] for entry in entries], dim=0).to(self.device)
        if not bool(valid.any()):
            raise ValueError("this batch has no valid flow cell; refusing to fit an empty mask")
        return reference, source, flow, valid

    def _update(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, Any]:
        reference, source, flow, valid = batch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        autocast = torch.amp.autocast(
            self.device.type, dtype=self._amp_dtype, enabled=self._use_amp
        )
        with autocast:
            output = self.model(reference, source, flow_gt=flow, iters=self.iters)
            loss = sequence_loss(
                output,
                flow,
                valid,
                gamma=float(self.config.optimizer.gamma),
                max_flow=float(self.config.optimizer.max_flow_packed),
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "the motion loss is not finite; stopping instead of poisoning weights"
            )
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_total = torch.nn.utils.clip_grad_norm_(
            self.params, float(self.config.optimizer.clip_grad)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        groups = self._gradient_groups()
        self.grad_evidence = {
            "last_total_norm": float(grad_total.detach())
            if torch.is_tensor(grad_total)
            else float(grad_total),
            "last_groups": groups,
        }
        return {"loss": float(loss.detach()), "grad_total": self.grad_evidence["last_total_norm"]}

    def _gradient_groups(self) -> dict[str, float]:
        groups: dict[str, float] = {name: 0.0 for name in GRADIENT_GROUPS}
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            group = name.split(".")[0]
            if group in groups:
                groups[group] += float(parameter.grad.detach().norm())
        return groups

    def advance(self, *, update: int, lr: float | None = None) -> dict[str, Any]:
        """One optimizer update: schedule the LR, draw the next batch, step the model."""
        step_lr = float(self.config.schedule.lr_at(update, self.base_lr) if lr is None else lr)
        self.note_lr(step_lr)
        self._apply_lr(step_lr)
        specs = [self.stream.next() for _ in range(int(self.config.data.batch_size))]
        self.batch_log.append([spec.as_dict() for spec in specs])
        record = self._update(self._batch_tensors(specs))
        self.train_losses.append(record["loss"])
        self.consumed_samples += len(specs)
        return {**record, "lr": step_lr, "samples": len(specs)}

    # -- schedule -------------------------------------------------------------------------

    def _apply_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def note_lr(self, lr: float) -> None:
        """Record one step's LR and count effective drops of the locked schedule."""
        self.lr_history.append(float(lr))
        if self._lr_reference is None or float(lr) > self._lr_reference:
            self._lr_reference = float(lr)
        elif float(lr) < self._lr_reference * (1.0 - float(self.config.schedule.min_drop_ratio)):
            self.lr_drops += 1
            self._lr_reference = float(lr)

    # -- diagnostics ----------------------------------------------------------------------

    def diagnose_fit(
        self, *, updates: int, specs: Sequence[MotionSampleSpec] | None = None
    ) -> dict[str, Any]:
        """Small-set fit / input / gradient diagnostics on a restored model clone.

        The model and optimizer are restored afterwards, so the diagnostic never
        moves the run's starting point.
        """
        updates = int(updates)
        if updates < 1:
            raise ValueError("diagnostic updates must be positive")
        fixed_specs = (
            tuple(specs)
            if specs is not None
            else self.train_specs[: self.config.diagnostics.fit_batch]
        )
        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        scaler_state = copy.deepcopy(self.scaler.state_dict())
        was_training = self.model.training
        losses: list[float] = []
        first_groups: dict[str, float] = {}
        try:
            batch = self._batch_tensors(fixed_specs)
            self._apply_lr(float(self.config.selection.lrs[0]))
            for step in range(updates):
                record = self._update(batch)
                losses.append(record["loss"])
                if step == 0:
                    first_groups = self._gradient_groups()
            inputs = torch.cat([batch[0], batch[1]], dim=0)
            self.diagnostic_samples += updates * len(fixed_specs)
        finally:
            self.model.load_state_dict(model_state)
            self.model.train(was_training)
            self.optimizer.load_state_dict(optimizer_state)
            self.scaler.load_state_dict(scaler_state)
        window = max(1, updates // 10)
        initial_mean = float(np.mean(losses[:window]))
        final_mean = float(np.mean(losses[-window:]))
        finite_inputs = bool(torch.isfinite(inputs).all())
        input_abs_max = float(inputs.abs().max()) if finite_inputs else float("inf")
        required = [name for name in GRADIENT_GROUPS if self._group_exists(name)]
        gradient_ok = all(
            math.isfinite(first_groups.get(name, 0.0)) and first_groups.get(name, 0.0) > 0.0
            for name in required
        )
        unit_probe = {
            "sensor_per_packed": self.sensor_per_packed,
            "sensor_px": [3.0, -5.0],
            "packed_cells": sensor_to_packed([3.0, -5.0], self.sensor_per_packed).tolist(),
        }
        unit_probe["round_trip_sensor_px"] = packed_to_sensor(
            unit_probe["packed_cells"], self.sensor_per_packed
        ).tolist()
        checks = {
            "small_set_fit_decreased": final_mean
            < initial_mean * float(self.config.diagnostics.fit_max_ratio),
            "small_set_fit_target": final_mean <= float(self.config.diagnostics.fit_target),
            "inputs_finite": finite_inputs,
            "inputs_within_limit": input_abs_max <= float(self.config.diagnostics.input_limit),
            "gradient_groups_nonzero": gradient_ok,
            "unit_round_trip_exact": bool(
                np.allclose(
                    unit_probe["round_trip_sensor_px"], unit_probe["sensor_px"], atol=0.0, rtol=0.0
                )
            ),
        }
        return {
            "updates": updates,
            "specs": [spec.as_dict() for spec in fixed_specs],
            "samples": updates * len(fixed_specs),
            "initial_window_mean": initial_mean,
            "final_window_mean": final_mean,
            "loss_curve": losses,
            "gradient_groups_first_step": first_groups,
            "required_gradient_groups": required,
            "input_abs_max": input_abs_max,
            "unit_probe": unit_probe,
            "checks": checks,
            "passed": all(checks.values()),
        }

    def _group_exists(self, group: str) -> bool:
        return any(
            name.split(".")[0] == group for name, _parameter in self.model.named_parameters()
        )

    # -- validation -----------------------------------------------------------------------

    def validate(self, *, update: int) -> dict[str, Any]:
        self.model.eval()
        losses: list[float] = []
        epes_packed: list[float] = []
        valid_cells = 0
        with torch.no_grad():
            for spec in self.validation_specs:
                reference, source, flow, valid = self._batch_tensors([spec])
                output = self.model(reference, source, flow_gt=flow, iters=self.iters)
                losses.append(
                    float(
                        sequence_loss(
                            output,
                            flow,
                            valid,
                            gamma=float(self.config.optimizer.gamma),
                            max_flow=float(self.config.optimizer.max_flow_packed),
                        )
                    )
                )
                predicted = output["final"]
                error = torch.sqrt(torch.sum((predicted - flow) ** 2, dim=1))
                mask = valid
                epes_packed.append(float((error * mask).sum() / mask.sum().clamp_min(1)))
                valid_cells += int(mask.sum())
        return {
            "update": int(update),
            "val_loss": float(np.mean(losses)),
            "val_epe_packed": float(np.mean(epes_packed)),
            "val_epe_sensor_px": float(np.mean(epes_packed) * self.sensor_per_packed),
            "pairs": len(self.validation_specs),
            "valid_cells": valid_cells,
        }

    # -- checkpointing --------------------------------------------------------------------

    def checkpoint_payload(
        self, *, update: int, platform: PlatformDecision | None = None
    ) -> dict[str, Any]:
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return {
            "schema": CHECKPOINT_SCHEMA,
            "update": int(update),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {
                "kind": "motion_absolute_drops",
                "base_lr": self.base_lr,
                "drops": self.config.schedule.as_list(),
                "update": int(update),
            },
            "scaler": self.scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": cuda_state,
            },
            "sampler": self.stream.state(),
            "train_samples": int(self.consumed_samples),
            "diagnostic_samples": int(self.diagnostic_samples),
            "samples_consumed": int(self.consumed_samples + self.diagnostic_samples),
            "best": copy.deepcopy(self.best),
            "val_history": copy.deepcopy(self.val_history),
            "lr_drops": int(self.lr_drops),
            "train_loss_tail": [float(value) for value in self.train_losses[-100:]],
            "batch_log_tail": copy.deepcopy(self.batch_log[-8:]),
            "seed": self.seed,
            "base_lr": self.base_lr,
            "iters": int(self.iters),
            "mode": self.mode,
            "depth": self.config.model.depth,
            "core_fingerprint": self.config.core_fingerprint,
            "hashes": {
                "source": _source_hashes(),
                "data_config": {
                    "path": self.config.data.config_path,
                    "sha256": _sha256_file(self.config.data.config_path),
                    "config_version": self.data_config.config_version,
                    "calibration_version": self.data_config.noise.calibration_version,
                    "noise_version": self.data_config.noise.noise_version,
                },
            },
            "diagnostics": copy.deepcopy(self.diagnostics),
            "platform": None if platform is None else platform.status,
            "parameter_hash": _parameter_digest(self.model),
        }

    def save_checkpoint(
        self, path: str | Path, *, update: int, platform: PlatformDecision | None = None
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(update=update, platform=platform), target)
        return target

    def load_checkpoint(self, path: str | Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not a {CHECKPOINT_SCHEMA} checkpoint")
        if payload.get("core_fingerprint") != self.config.core_fingerprint:
            raise ValueError(
                "the checkpoint was written under a different run definition (model/data/"
                "schedule/optimizer/selection); start a new run instead of continuing this one"
            )
        if int(payload.get("seed", -1)) != self.seed:
            raise ValueError("the checkpoint seed differs from this run's seed")
        if int(payload.get("iters", -1)) != self.iters:
            raise ValueError("the checkpoint iteration count differs from this run's iters")
        if abs(float(payload.get("base_lr", -1.0)) - self.base_lr) > 0.0:
            raise ValueError("the checkpoint base learning rate differs from this run's")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if torch.cuda.is_available() and payload["rng"].get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        self.stream.load_state(payload["sampler"])
        self.consumed_samples = int(
            payload.get("train_samples", payload.get("samples_consumed", 0))
        )
        self.diagnostic_samples = int(payload.get("diagnostic_samples", 0))
        self.best = payload.get("best")
        self.val_history = list(payload.get("val_history", []))
        self.lr_drops = int(payload.get("lr_drops", 0))
        self._lr_reference = None
        self.diagnostics = payload.get("diagnostics")
        self._param_before = self._parameter_snapshot()
        return int(payload["update"])


# ---------------------------------------------------------------------------
# run modes
# ---------------------------------------------------------------------------


def _diagnostics_or_previous(trainer: MotionTrainer) -> dict[str, Any]:
    if trainer.diagnostics is not None:
        return trainer.diagnostics
    return trainer.diagnose_fit(updates=int(trainer.config.diagnostics.fit_updates))


def _assert_region_discipline(trainer: MotionTrainer) -> None:
    """Refuse a run whose regions or policy break the four-way split contract."""
    if not trainer.validation_specs:
        raise ValueError("the run needs a non-empty validation region")
    for spec in trainer.train_specs:
        if spec.split != "train":
            raise ValueError(f"a training spec declares split {spec.split!r}, not 'train'")
    for spec in trainer.validation_specs:
        if spec.split != "validation":
            raise ValueError(f"a validation spec declares split {spec.split!r}, not 'validation'")
    policy = trainer.data_config.policy
    if policy.selection_split != "validation":
        raise ValueError(
            f"the motion data policy selects on {policy.selection_split!r}; the platform contract "
            "requires motion validation"
        )
    if policy.evaluation_split != "test":
        raise ValueError("the motion data policy must keep 'test' as the evaluation region")
    declared = tuple(policy.fitting_splits)
    if declared != ("train", "calibration"):
        raise ValueError(
            f"the motion data policy fits on {declared}; expected ('train', 'calibration')"
        )


def _resolve_config_path(config_path: str | Path | None) -> Path:
    """The config file a run was started from; recorded in every receipt."""
    if config_path is None:
        return Path("configs/encoders/n01_full/p2_motion_train.json")
    return Path(config_path)


def run_fit_check(
    config: MotionTrainingConfig,
    *,
    output: str | Path,
    device: str | torch.device = "cpu",
    max_updates: int | None = None,
    seed: int | None = None,
    lr: float | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """``fit-check``: can the author loss drive a small set to a lower loss?"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    updates = int(config.diagnostics.check_updates if max_updates is None else max_updates)
    trainer = MotionTrainer(config, device=device, seed=seed, lr=lr)
    _assert_region_discipline(trainer)
    started = time.perf_counter()
    diagnostics = trainer.diagnose_fit(updates=updates)
    receipt = {
        "run_mode": "fit-check",
        "schema": SCHEMA,
        "git_sha": git_sha(),
        "config": config.as_dict(),
        "config_fingerprint": config.core_fingerprint,
        "config_path": str(_resolve_config_path(config_path)),
        "run_scale": config.scale,
        "device": str(trainer.device),
        "seed": trainer.seed,
        "base_lr": trainer.base_lr,
        "iters": trainer.iters,
        "max_updates": updates,
        "diagnostics": diagnostics,
        "passed": bool(diagnostics["passed"]),
        "samples_consumed": int(diagnostics["samples"]),
        "parameter_hash_after": _parameter_digest(trainer.model),
        "input_hashes": _source_hashes(_resolve_config_path(config_path)),
        "data_hashes": {
            "data_config_sha256": _sha256_file(config.data.config_path),
            "calibration_version": trainer.data_config.noise.calibration_version,
            "noise_version": trainer.data_config.noise.noise_version,
        },
        "wall_seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "python": __import__("sys").version,
        "numpy": np.__version__,
    }
    (output / "fit_check.json").write_text(
        json.dumps(_jsonable(receipt), indent=2, default=str) + "\n"
    )
    return receipt


def run_selection(
    config: MotionTrainingConfig,
    *,
    output: str | Path,
    device: str | torch.device = "cpu",
    max_updates: int | None = None,
    seed: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """``select``: two LR tiers x iteration tiers at one budget, chosen on validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    updates = int(config.selection.updates if max_updates is None else max_updates)
    candidates: list[dict[str, Any]] = []
    for lr in config.selection.lrs:
        for iters in config.selection.iters:
            trainer = MotionTrainer(config, device=device, seed=seed, lr=lr, iters=iters)
            _assert_region_discipline(trainer)
            started = time.perf_counter()
            for update in range(updates):
                trainer.advance(update=update)
            metrics = trainer.validate(update=updates)
            candidates.append(
                {
                    "lr": float(lr),
                    "iters": int(iters),
                    "updates": updates,
                    "val_loss": metrics["val_loss"],
                    "val_epe_packed": metrics["val_epe_packed"],
                    "val_epe_sensor_px": metrics["val_epe_sensor_px"],
                    "train_loss_first": trainer.train_losses[0],
                    "train_loss_last": trainer.train_losses[-1],
                    "parameter_update": trainer.parameter_update_evidence(),
                    "samples_consumed": trainer.consumed_samples,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    winner = min(candidates, key=lambda item: (item["val_epe_sensor_px"], item["val_loss"]))
    receipt = {
        "run_mode": "select",
        "schema": SCHEMA,
        "git_sha": git_sha(),
        "config_fingerprint": config.core_fingerprint,
        "config_path": str(_resolve_config_path(config_path)),
        "run_scale": config.scale,
        "device": str(device),
        "seed": int(config.seeds[0] if seed is None else seed),
        "budget_per_candidate": updates,
        "same_budget": True,
        "selection_split": "validation",
        "fitting_split": "train",
        "test_split_used": False,
        "candidates": candidates,
        "winner": {"lr": winner["lr"], "iters": winner["iters"]},
        "selection_metric": "(val_epe_sensor_px, val_loss)",
        "input_hashes": _source_hashes(_resolve_config_path(config_path)),
        "wall_seconds": sum(item["wall_seconds"] for item in candidates),
    }
    (output / "selection.json").write_text(
        json.dumps(_jsonable(receipt), indent=2, default=str) + "\n"
    )
    (output / "winner.json").write_text(
        json.dumps({"winner": receipt["winner"], "selection_split": "validation"}, indent=2) + "\n"
    )
    return receipt


def run_training(
    config: MotionTrainingConfig,
    *,
    output: str | Path,
    device: str | torch.device = "cpu",
    resume: str | Path | None = None,
    max_updates: int | None = None,
    seed: int | None = None,
    lr: float | None = None,
    iters: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """``train``: formal budget with the contract's platform criteria and receipts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    budget = int(config.budget.updates if max_updates is None else max_updates)
    if budget > int(config.budget.max_updates):
        raise ValueError(
            f"requested {budget} updates exceeds the declared ceiling {config.budget.max_updates}"
        )
    trainer = MotionTrainer(config, device=device, seed=seed, lr=lr, iters=iters)
    _assert_region_discipline(trainer)
    started = time.perf_counter()
    if resume is not None:
        trainer.start_update = trainer.load_checkpoint(resume)
        if trainer.start_update >= budget:
            platform = decide_platform(
                val_history=trainer.val_history,
                budget=budget,
                interval=config.budget.validation_interval,
                monitor=config.budget.monitor,
                delta_val=float(config.budget.delta_val),
                plateau_checks=int(config.budget.plateau_checks),
                lr_drops=trainer.lr_drops,
                final_phase=config.schedule.is_final_phase(max(0, budget - 1)),
                diagnostics_passed=bool((trainer.diagnostics or {}).get("passed", False)),
            )
            return _finish_train(
                trainer,
                output=output,
                checkpoint_dir=checkpoint_dir,
                budget=budget,
                platform=platform,
                started=started,
                config_path=_resolve_config_path(config_path),
            )
    trainer.diagnostics = _diagnostics_or_previous(trainer)
    interval = int(config.budget.validation_interval)
    platform: PlatformDecision | None = None
    completed_update = int(trainer.start_update)
    for update in range(trainer.start_update, budget):
        trainer.advance(update=update)
        completed = update + 1
        completed_update = completed
        is_last = completed == budget
        if completed % interval == 0 or is_last:
            metrics = trainer.validate(update=completed)
            trainer.val_history.append(metrics)
            if trainer.best is None or (
                metrics[config.budget.monitor] < trainer.best[config.budget.monitor]
            ):
                trainer.best = dict(metrics)
                trainer.save_checkpoint(checkpoint_dir / "best.pt", update=completed)
            platform = decide_platform(
                val_history=trainer.val_history,
                budget=budget,
                interval=interval,
                monitor=config.budget.monitor,
                delta_val=float(config.budget.delta_val),
                plateau_checks=int(config.budget.plateau_checks),
                lr_drops=trainer.lr_drops,
                final_phase=config.schedule.is_final_phase(update),
                diagnostics_passed=bool(trainer.diagnostics["passed"]),
            )
        if completed % int(config.checkpoint.every) == 0 or is_last:
            trainer.save_checkpoint(checkpoint_dir / "last.pt", update=completed, platform=platform)
        if platform is not None and platform.status == "validation_plateau":
            trainer.save_checkpoint(checkpoint_dir / "last.pt", update=completed, platform=platform)
            break
    if platform is None:
        platform = decide_platform(
            val_history=trainer.val_history,
            budget=budget,
            interval=interval,
            monitor=config.budget.monitor,
            delta_val=float(config.budget.delta_val),
            plateau_checks=int(config.budget.plateau_checks),
            lr_drops=trainer.lr_drops,
            final_phase=config.schedule.is_final_phase(max(0, budget - 1)),
            diagnostics_passed=bool(trainer.diagnostics["passed"]),
        )
    trainer.save_checkpoint(checkpoint_dir / "last.pt", update=completed_update, platform=platform)
    return _finish_train(
        trainer,
        output=output,
        checkpoint_dir=checkpoint_dir,
        budget=budget,
        platform=platform,
        started=started,
        config_path=_resolve_config_path(config_path),
    )


def _finish_train(
    trainer: MotionTrainer,
    *,
    output: Path,
    checkpoint_dir: Path,
    budget: int,
    platform: PlatformDecision,
    started: float,
    config_path: Path,
) -> dict[str, Any]:
    trainer.wall_seconds = time.perf_counter() - started
    losses = trainer.train_losses
    stopped_at = int(trainer.start_update) + len(losses)
    window = max(1, min(100, len(losses)))
    summary = {
        "updates": len(losses),
        "first": float(losses[0]) if losses else None,
        "last": float(losses[-1]) if losses else None,
        "first_window_mean": float(np.mean(losses[:window])) if losses else None,
        "last_window_mean": float(np.mean(losses[-window:])) if losses else None,
        "min": float(np.min(losses)) if losses else None,
    }
    receipt = {
        "run_mode": "train",
        "schema": SCHEMA,
        "git_sha": git_sha(),
        "config": trainer.config.as_dict(),
        "config_fingerprint": trainer.config.core_fingerprint,
        "run_scale": trainer.config.scale,
        "device": str(trainer.device),
        "amp": trainer._amp,
        "seed": trainer.seed,
        "base_lr": trainer.base_lr,
        "iters": trainer.iters,
        "mode": trainer.mode,
        "depth": trainer.config.model.depth,
        "start_update": trainer.start_update,
        "budget": int(budget),
        "platform": {
            "status": platform.status,
            "reasons": list(platform.reasons),
            "details": _jsonable(platform.details),
            "stopped_at_update": stopped_at,
            "criteria": {
                "validation_interval": int(trainer.config.budget.validation_interval),
                "delta_val": float(trainer.config.budget.delta_val),
                "monitor": trainer.config.budget.monitor,
                "plateau_checks": int(trainer.config.budget.plateau_checks),
                "min_lr_drops": 2,
                "lr_drops_observed": int(trainer.lr_drops),
                "final_low_lr_phase": bool(
                    trainer.config.schedule.is_final_phase(max(0, stopped_at - 1))
                ),
            },
        },
        "loss_curve_summary": summary,
        "val_history": _jsonable(trainer.val_history),
        "best": _jsonable(trainer.best),
        "lr_drops": int(trainer.lr_drops),
        "optimizer_updates": len(losses),
        "samples_consumed": int(trainer.consumed_samples + trainer.diagnostic_samples),
        "diagnostic_samples": int(trainer.diagnostic_samples),
        "samples_per_update": int(trainer.config.data.batch_size),
        "parameter_updates": trainer.parameter_update_evidence(),
        "gradient_evidence": _jsonable(trainer.grad_evidence),
        "diagnostics": _jsonable(trainer.diagnostics),
        "provider": trainer.provider.receipt(),
        "sampler": trainer.stream.receipt(),
        "batch_identity_head": trainer.batch_log[:4],
        "input_hashes": _source_hashes(config_path),
        "data_hashes": {
            "data_config_sha256": _sha256_file(trainer.config.data.config_path),
            "calibration_version": trainer.data_config.noise.calibration_version,
            "noise_version": trainer.data_config.noise.noise_version,
            "normalization_versions": sorted(trainer.normalization),
        },
        "checkpoints": {
            "last": str(checkpoint_dir / "last.pt"),
            "best": str(checkpoint_dir / "best.pt") if trainer.best is not None else None,
        },
        "parameter_hash": _parameter_digest(trainer.model),
        "wall_seconds": trainer.wall_seconds,
        "torch": torch.__version__,
        "python": __import__("sys").version,
        "numpy": np.__version__,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if trainer.device.type == "cuda" else None
        ),
        "notes": trainer.config.notes,
    }
    (output / "train_loss_curve.json").write_text(
        json.dumps(
            {
                "updates": list(
                    range(trainer.start_update + 1, trainer.start_update + 1 + len(losses))
                ),
                "loss": [float(value) for value in losses],
                "lr": [float(value) for value in trainer.lr_history],
            }
        ),
    )
    (output / "receipt.json").write_text(
        json.dumps(_jsonable(receipt), indent=2, default=str) + "\n"
    )
    return receipt


def apply_selection(
    trainer_config: MotionTrainingConfig, selection_path: str | Path
) -> dict[str, Any]:
    """Read a ``select`` winner; the formal run must use the validation-selected setting."""
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    winner = payload.get("winner")
    if not isinstance(winner, Mapping) or "lr" not in winner or "iters" not in winner:
        raise ValueError(f"{selection_path} does not carry a selection winner")
    if payload.get("selection_split") != "validation":
        raise ValueError("the selection receipt was not decided on the validation region")
    if payload.get("test_split_used") is not False:
        raise ValueError("the selection receipt does not certify that test was unused")
    if payload.get("config_fingerprint") != trainer_config.core_fingerprint:
        raise ValueError("the selection receipt belongs to a different run definition")
    return {"lr": float(winner["lr"]), "iters": int(winner["iters"])}


# ---------------------------------------------------------------------------
# stem report: the two stems must differ only by their input channels
# ---------------------------------------------------------------------------

#: Top-level trainable modules of the vendored motion network, in model order.
STEM_GROUPS = ("cnet", "fnet", "init_conv", "update_block", "flow_head", "upsample_weight")
#: The only input-channel-dependent layer: ``Conv2d(C, 64, kernel_size=7, stride=2, padding=3)``
#: at the head of each ``ResNetFPN`` (``fnet`` takes the RAW channels, ``cnet`` their 2x concat).
STEM_CONV_KERNEL = 7
STEM_CONV_OUT_CHANNELS = 64

#: Configuration paths that may differ between the two stem arms, each for a
#: declared reason (label, stem domain, input geometry/resolution).  Anything
#: else differing is a hidden difference and is refused.
STEM_CONFIG_EXPECTED_DIFFERENCES = (
    "scale",
    "stage",
    "notes",
    "model.mode",
    "data.input_mode",
    "data.geometry",
    "data.recipe_scale",
)


def model_parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """Per-group trainable parameters and their names/shapes for one model."""
    groups: dict[str, Any] = {
        name: {"params": 0, "tensor_count": 0, "tensors": {}} for name in STEM_GROUPS
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = name.split(".")[0]
        if group not in groups:
            raise RuntimeError(
                f"unexpected trainable parameter group {group!r} for {name!r}; the stem report "
                "only knows the declared motion-network groups"
            )
        groups[group]["params"] += int(parameter.numel())
        groups[group]["tensor_count"] += 1
        groups[group]["tensors"][name] = list(parameter.shape)
    groups["total"] = {
        "params": sum(groups[name]["params"] for name in STEM_GROUPS),
        "tensor_count": sum(groups[name]["tensor_count"] for name in STEM_GROUPS),
        "tensors": {},
    }
    return groups


def analytic_stem_delta(channels_from: int, channels_to: int) -> dict[str, int]:
    """Expected parameter delta from switching the stem input channel count.

    Only the two stem convolutions see the RAW channel count: ``fnet`` directly
    and ``cnet`` through the ``2 * C`` concatenation.  Every other group keeps
    exactly the same shapes, so its expected delta is zero.  The values are
    compared against the measured difference in :func:`run_stem_report`.
    """
    for value in (channels_from, channels_to):
        if int(value) < 1:
            raise ValueError(f"a stem needs a positive channel count, got {value!r}")
    weights = STEM_CONV_OUT_CHANNELS * STEM_CONV_KERNEL * STEM_CONV_KERNEL
    delta = {name: 0 for name in STEM_GROUPS}
    delta["fnet"] = (int(channels_to) - int(channels_from)) * weights
    delta["cnet"] = (2 * int(channels_to) - 2 * int(channels_from)) * weights
    return delta


def _matches_expected_path(declared: str, path: str) -> bool:
    """Whether a flattened config path belongs to one declared path (nested included)."""
    return path == declared or path.startswith(f"{declared}.")


def _flatten_config(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_config(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def run_stem_report(
    config: MotionTrainingConfig,
    compare: MotionTrainingConfig,
    *,
    output: str | Path,
    device: str | torch.device = "cpu",
    config_path: str | Path | None = None,
    compare_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Report and verify the CFA4 vs pseudo-RAW RGB3 stem parameter difference.

    Both models are built from their own configuration.  The report proves that
    the only parameter difference comes from the stem input channels (4 -> 3)
    and their concatenation in ``cnet`` (8 -> 6): depths and iteration counts
    must match, the set of trainable parameter names must match, the differing
    parameter names must be exactly the two stem convolutions, and the measured
    per-group deltas must equal :func:`analytic_stem_delta`.  The configuration
    difference is checked against an explicit expected list, so a hidden change
    to readout, depth, budget or criteria is refused rather than reported.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if config.model.depth != compare.model.depth:
        raise ValueError(
            "hidden difference: the two arms use different model depths "
            f"({config.model.depth!r} vs {compare.model.depth!r}); the stem report compares stems, "
            "not architectures"
        )
    if int(config.model.iters) != int(compare.model.iters):
        raise ValueError(
            "hidden difference: the two arms use different iteration counts "
            f"({config.model.iters} vs {compare.model.iters})"
        )
    left_flat = _flatten_config(config.as_dict())
    right_flat = _flatten_config(compare.as_dict())
    observed_differences = tuple(
        sorted(
            key
            for key in set(left_flat) | set(right_flat)
            if left_flat.get(key) != right_flat.get(key)
        )
    )

    # a declared path also covers its nested paths (e.g. "data.geometry.height"
    # when one arm declares a geometry override and the other uses the shipped one)
    unexpected = tuple(
        key
        for key in observed_differences
        if not any(
            _matches_expected_path(expected, key) for expected in STEM_CONFIG_EXPECTED_DIFFERENCES
        )
    )
    if unexpected:
        raise ValueError(
            f"hidden configuration difference beyond the declared stem switch: {unexpected}"
        )
    missing = tuple(
        expected
        for expected in STEM_CONFIG_EXPECTED_DIFFERENCES
        if not any(_matches_expected_path(expected, key) for key in observed_differences)
    )
    if missing:
        raise ValueError(
            "the two arms were expected to differ on "
            f"{missing}; they are not a CFA vs RGB3 stem pair"
        )

    geometries = {}
    estimates = {}
    for label, arm in (("cfa4", config), ("rgb3", compare)):
        data_config = arm.data.resolved_data_config()
        input_mode = arm.data.input_mode or data_config.geometry.input_mode
        geometry = data_config.geometry_for(input_mode)
        geometries[label] = {
            "input_mode": input_mode,
            "mode": arm.model.mode,
            "is_cfa": bool(geometry.is_cfa),
            "packed_shape": list(geometry.packed_shape),
            "sensor_shape": list(geometry.sensor_shape),
            "channel_phase": list(geometry.channel_phase),
            "cfa_pattern": geometry.cfa_pattern,
            "sensor_per_packed": int(SENSOR_PER_PACKED["cfa4" if geometry.is_cfa else "rgb3"]),
            "depth": arm.model.depth,
            "iters": int(arm.model.iters),
        }
        estimator = build_estimator(arm, data_config, device=device)
        estimates[label] = estimator.model
    reports = {label: model_parameter_report(model) for label, model in estimates.items()}
    channels = {
        "cfa4": int(geometries["cfa4"]["packed_shape"][0]) and 4,
        "rgb3": 3,
    }
    # channel counts come from the stems themselves, not from the labels
    channels = {
        label: int(estimates[label].state_dict()["fnet.conv1.weight"].shape[1])
        for label in ("cfa4", "rgb3")
    }
    analytic = analytic_stem_delta(channels["cfa4"], channels["rgb3"])
    shape_index = {
        label: {
            name: tuple(shape)
            for group in STEM_GROUPS
            for name, shape in reports[label][group]["tensors"].items()
        }
        for label in ("cfa4", "rgb3")
    }
    # the parameter *names* are identical; the hidden-difference guard is that the
    # shapes differ on exactly the two stem convolutions and nowhere else
    differing = tuple(
        sorted(
            name
            for name in set(shape_index["cfa4"]) | set(shape_index["rgb3"])
            if shape_index["cfa4"].get(name) != shape_index["rgb3"].get(name)
        )
    )
    groups: dict[str, Any] = {}
    analytic_matches = True
    for group in STEM_GROUPS:
        measured = reports["rgb3"][group]["params"] - reports["cfa4"][group]["params"]
        expected = int(analytic[group])
        shapes_equal = reports["cfa4"][group]["tensors"] == reports["rgb3"][group]["tensors"]
        differing_tensors = {
            name: {
                "cfa4": reports["cfa4"][group]["tensors"].get(name),
                "rgb3": reports["rgb3"][group]["tensors"].get(name),
            }
            for name in sorted(
                set(reports["cfa4"][group]["tensors"]) | set(reports["rgb3"][group]["tensors"])
            )
            if reports["cfa4"][group]["tensors"].get(name)
            != reports["rgb3"][group]["tensors"].get(name)
        }
        groups[group] = {
            "cfa4_params": reports["cfa4"][group]["params"],
            "rgb3_params": reports["rgb3"][group]["params"],
            "delta_params": measured,
            "analytic_delta_params": expected,
            "matches_analytic": measured == expected,
            "tensor_count": reports["cfa4"][group]["tensor_count"],
            "shapes_equal": shapes_equal,
            "differing_tensors": differing_tensors,
        }
        analytic_matches = analytic_matches and measured == expected
    total = {
        "cfa4": reports["cfa4"]["total"]["params"],
        "rgb3": reports["rgb3"]["total"]["params"],
        "delta": reports["rgb3"]["total"]["params"] - reports["cfa4"]["total"]["params"],
        "analytic_delta": sum(analytic.values()),
        "matches_analytic": reports["rgb3"]["total"]["params"] - reports["cfa4"]["total"]["params"]
        == sum(analytic.values()),
    }
    only_stem = differing == ("cnet.conv1.weight", "fnet.conv1.weight")
    if not only_stem:
        raise RuntimeError(
            "hidden difference: the trainable parameter shapes differ beyond the two stem "
            f"convolutions: {differing}"
        )
    if not (analytic_matches and total["matches_analytic"]):
        raise RuntimeError(
            f"the measured stem deltas disagree with the analytic expectation: "
            f"analytic={analytic}, total={total}"
        )
    receipt = {
        "run_mode": "stem-report",
        "schema": SCHEMA,
        "git_sha": git_sha(),
        "device": str(device),
        "scale": {"cfa4": config.scale, "rgb3": compare.scale},
        "configs": {
            "cfa4": {
                "path": str(_resolve_config_path(config_path)),
                "sha256": _sha256_file(_resolve_config_path(config_path))
                if _resolve_config_path(config_path).is_file()
                else None,
                "core_fingerprint": config.core_fingerprint,
            },
            "rgb3": {
                "path": str(_resolve_config_path(compare_config_path)),
                "sha256": _sha256_file(_resolve_config_path(compare_config_path))
                if _resolve_config_path(compare_config_path).is_file()
                else None,
                "core_fingerprint": compare.core_fingerprint,
            },
        },
        "geometries": geometries,
        "channels": channels,
        "stem_conv": {
            "kernel": [STEM_CONV_KERNEL, STEM_CONV_KERNEL],
            "out_channels": STEM_CONV_OUT_CHANNELS,
            "fnet_input": [channels["cfa4"], channels["rgb3"]],
            "cnet_input": [2 * channels["cfa4"], 2 * channels["rgb3"]],
        },
        "total_trainable": total,
        "groups": groups,
        "differing_parameter_names": list(differing),
        "only_stem_input_channels_differ": bool(only_stem),
        "config_difference": {
            "expected_paths": list(STEM_CONFIG_EXPECTED_DIFFERENCES),
            "observed_paths": list(observed_differences),
            "matches_expected": True,
        },
        "verdict": {
            "only_expected_config_differences": True,
            "only_stem_parameters_differ": bool(only_stem),
            "analytic_matches_measured": bool(analytic_matches and total["matches_analytic"]),
            "evidence_scale": {
                "cfa4": config.scale,
                "rgb3": compare.scale,
                "statement": (
                    "the cfa4 arm carries the authorised formal budget; the rgb3 arm is a "
                    "development-scale arm (same budget/schedule/criteria, dev-scale evidence)"
                ),
            },
        },
    }
    (output / "stem_report.json").write_text(
        json.dumps(_jsonable(receipt), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return receipt


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N01 v1.0 P2 RAW motion training")
    parser.add_argument("mode", choices=("fit-check", "select", "stem-report", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--iters", type=int)
    parser.add_argument(
        "--compare",
        type=Path,
        help="the second motion training config; required by the stem-report mode",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help=(
            "selection.json produced by the select stage; applies the validation-selected "
            "LR and iteration tier"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_motion_training_config(args.config)
    lr = args.lr
    iters = args.iters
    if args.selection is not None:
        if args.mode != "train":
            raise ValueError("--selection only applies to the train mode")
        winner = apply_selection(config, args.selection)
        lr = winner["lr"] if lr is None else lr
        iters = winner["iters"] if iters is None else iters
    if args.mode == "fit-check":
        receipt = run_fit_check(
            config,
            output=args.output,
            device=args.device,
            max_updates=args.max_updates,
            seed=args.seed,
            lr=lr,
            config_path=args.config,
        )
        print(json.dumps({"run_mode": "fit-check", "passed": receipt["passed"]}))
        return 0 if receipt["passed"] else 1
    if args.mode == "stem-report":
        if args.compare is None:
            raise ValueError("the stem-report mode needs --compare <rgb3 config>")
        compare = load_motion_training_config(args.compare)
        receipt = run_stem_report(
            config,
            compare,
            output=args.output,
            device=args.device,
            config_path=args.config,
            compare_config_path=args.compare,
        )
        print(
            json.dumps(
                {
                    "run_mode": "stem-report",
                    "total_trainable_delta": receipt["total_trainable"]["delta"],
                    "only_stem_input_channels_differ": receipt["only_stem_input_channels_differ"],
                }
            )
        )
        return 0
    if args.mode == "select":
        receipt = run_selection(
            config,
            output=args.output,
            device=args.device,
            max_updates=args.max_updates,
            seed=args.seed,
            config_path=args.config,
        )
        print(json.dumps({"run_mode": "select", "winner": receipt["winner"]}))
        return 0
    receipt = run_training(
        config,
        output=args.output,
        device=args.device,
        resume=args.resume,
        max_updates=args.max_updates,
        seed=args.seed,
        lr=lr,
        iters=iters,
        config_path=args.config,
    )
    print(
        json.dumps(
            {
                "run_mode": "train",
                "platform": receipt["platform"]["status"],
                "updates": receipt["optimizer_updates"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

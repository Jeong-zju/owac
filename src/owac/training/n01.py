"""Training CLI for the N01 adaptive spatio-temporal evidence integrator.

Step 4 of ``docs/prompts/DeepSeek_N01_implementation.md``.  This module owns the
*online* training protocol and the shared mechanism probe head; it never
reimplements the encoder.  Boundaries that matter:

* Frames go through exactly the same :meth:`N01Encoder.step` used at inference.
  The measurement buffer and the support graph are advanced one frame at a time;
  gradients are truncated at ``--tbptt`` boundaries with :func:`state.detach` and
  a plain random batch resets the whole state at the window start.
* ``L = w_loc*L_location + w_meas*L_measurement_prediction + w_budget*L_update_cost``
  with per-term weights from ``--loss-weights`` and per-term valid-sample counts
  logged separately.
* The measurement-prediction target is a *held-out* pixel block of an earlier
  frame: those pixels are replaced in the input by a neighbourhood mean and
  marked invalid, so the input can never bypass-copy its own target.  Pixels that
  are both input and target are therefore excluded by construction, and the
  exclusion is reported in ``run.json``.
* ``lambda_budget`` defaults to 0.  A non-zero value adds a cost compared against
  a declared target update count, and the task error stays in the log next to it.

Every number written to ``run.json`` is measured: peak RSS from
``resource.getrusage`` (``null`` when unavailable), peak GPU memory from
``torch.cuda.max_memory_allocated()`` (``null`` on CPU), per-step wall time
percentiles from real timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by the CLI smoke test, not by unit tests
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "owac.training.n01 needs PyTorch, which is the optional 'encoder' extra.\n"
        "Install it with: uv sync --locked --extra encoder\n"
        f"Original import error: {exc}"
    ) from exc

from owac.core.encoder_contract import ComputeBudget, EncoderOutput, MeasurementFrame

__all__ = [
    "LocationProbeHead",
    "MeasurementPredictionHead",
    "N01ProbeHeads",
    "SyntheticSource",
    "TrainingSettings",
    "SequenceStepResult",
    "batch_times",
    "build_probe_heads",
    "checkpoint_state",
    "evaluate_sequence",
    "git_sha",
    "load_checkpoint",
    "loss_weights_from_args",
    "main",
    "make_generator",
    "measurement_holdout",
    "peak_rss_bytes",
    "frame_variance_metadata",
    "probe_parameter_count",
    "save_checkpoint",
    "sequence_step",
    "tensor_bytes",
    "train",
    "train_one_step",
    "units_note",
]

#: Pixel-block period of the measurement holdout.  Period 2 with offset 0 keeps
#: one quarter of the pixels of the target frame as the prediction target.
_HOLDOUT_PERIOD = 2
_HOLDOUT_OFFSETS = (0, 1)
_DEFAULT_TARGET_UPDATES = 1.0
_EPS = 1e-8


class _OptionalDependencyError(SystemExit):
    """Raised when a parallel-implementation dependency is not readable yet."""


def _require(module_path: str, purpose: str) -> Any:
    """Import a module that another implementer owns, with a readable failure."""
    import importlib

    try:
        return importlib.import_module(module_path)
    except ImportError as exc:  # pragma: no cover - depends on parallel work
        raise _OptionalDependencyError(
            f"{purpose} needs '{module_path}', which is not importable yet "
            f"({exc}). This module is implemented in parallel; rerun once it lands."
        ) from exc


# ---------------------------------------------------------------------------
# units and small bookkeeping helpers
# ---------------------------------------------------------------------------
def units_note() -> dict[str, str]:
    """Explicit unit declaration for every number this CLI reports."""
    return {
        "location_heatmap": (
            "logits over the HxW measurement grid; cross-entropy target is a Gaussian "
            "heatmap with sigma in pixels"
        ),
        "location_huber": "pixel units: mean over the diagonal of |pred_xy - target_xy|",
        "position_error": "pixels, row/col convention matching target_xy (normalised in [0,1])",
        "measurement_prediction": (
            "normalised measurement units (x=(raw-black)/(white-black)); target is the "
            "held-out pixel block of frame t-1, input replacement is a neighbourhood mean"
        ),
        "budget": "effective update count per step minus the declared target update count",
        "latency_ms": (
            "milliseconds of wall time on this machine; the sample age (algorithmic lag) is "
            "reported separately and is never mixed into this number"
        ),
        "memory_bytes": "bytes",
    }


def measurement_holdout(step: int) -> tuple[int, int]:
    """Deterministic holdout block for ``step``: (row offset, col offset) mod 2.

    Pixels with ``(row % 2, col % 2) == (row_offset, col_offset)`` of the target
    frame are the measurement-prediction target.  The same pixels are replaced in
    the input, so a pixel is never simultaneously an input and its own target.
    """
    index = int(step) % (len(_HOLDOUT_OFFSETS) ** 2)
    return _HOLDOUT_OFFSETS[index // len(_HOLDOUT_OFFSETS)], _HOLDOUT_OFFSETS[
        index % len(_HOLDOUT_OFFSETS)
    ]


def holdout_mask(height: int, width: int, step: int, device: Any, dtype: Any) -> Any:
    row_offset, col_offset = measurement_holdout(step)
    rows = torch.arange(height, device=device).view(-1, 1) % _HOLDOUT_PERIOD == row_offset
    cols = torch.arange(width, device=device).view(1, -1) % _HOLDOUT_PERIOD == col_offset
    mask = rows & cols
    if mask.shape != (height, width):  # pragma: no cover - defensive
        raise RuntimeError("holdout mask shape mismatch")
    return mask.to(dtype=dtype)


def _mean_filter(values: Any, radius: int = 1) -> Any:
    kernel = 2 * radius + 1
    return F.avg_pool2d(
        values, kernel_size=kernel, stride=1, padding=radius, count_include_pad=False
    )


def tensor_bytes(tensor: Any) -> int:
    if tensor is None or not hasattr(tensor, "numel"):
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def peak_rss_bytes() -> int | None:
    """Measured peak RSS of this process; ``None`` when the platform has no ru_maxrss."""
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        return None
    if value <= 0:
        return None
    # Linux reports kibibytes, macOS bytes.
    return value * 1024 if sys.platform != "darwin" else value


def peak_gpu_bytes() -> int | None:
    """Measured peak allocated GPU memory, or ``None`` when no CUDA device is used."""
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated())
    except RuntimeError:  # pragma: no cover - defensive
        return None


def git_sha(repo_root: Path | None = None) -> str:
    cwd = str(repo_root) if repo_root is not None else None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def make_generator(seed: int | None) -> Any:
    generator = torch.Generator()
    generator.manual_seed(int(seed) if seed is not None else int(torch.seed()))
    return generator


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # A dataclass with no tensor payload (a condition descriptor, a settings
    # record) serialises field by field instead of failing the whole write.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# shared mechanism probe head
# ---------------------------------------------------------------------------
class LocationProbeHead(nn.Module):
    """The one localisation head every mechanism probe reads out through.

    ``encode`` returns a feature map ``[B, C_feat, H, W]``; this head turns it
    into a distribution over the ``H*W`` measurement grid.  The same head,
    optimiser, step count and data are used for N01 and for every baseline in the
    comparison suite, which is what makes the budget comparison controlled.
    """

    def __init__(self, in_channels: int, hidden: int = 32, kernel: int = 3):
        super().__init__()
        padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, features: Any, size: tuple[int, int] | None = None) -> Any:
        """Return ``[B, H, W]`` logits at the measurement resolution."""
        logits = self.net(features)
        if size is not None and logits.shape[-2:] != tuple(size):
            logits = F.interpolate(logits, size=tuple(size), mode="bilinear", align_corners=False)
        return logits.squeeze(1)

    @staticmethod
    def heatmap_target(
        target_xy: Any, size: tuple[int, int], sigma: float, dtype: Any = torch.float32
    ) -> Any:
        """Gaussian supervision target in pixel units, normalised to sum to 1."""
        height, width = size
        device = target_xy.device
        rows = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
        cols = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
        centre_y = target_xy[:, 0].to(dtype).clamp(0.0, 1.0) * (height - 1)
        centre_x = target_xy[:, 1].to(dtype).clamp(0.0, 1.0) * (width - 1)
        distance = (rows - centre_y.view(-1, 1, 1)) ** 2 + (cols - centre_x.view(-1, 1, 1)) ** 2
        heat = torch.exp(-0.5 * distance / (sigma * sigma))
        return heat / heat.sum(dim=(1, 2), keepdim=True).clamp_min(_EPS)

    @staticmethod
    def soft_argmax(logits: Any) -> Any:
        """Expected (row, col) in pixels from logits over ``[H, W]``."""
        batch, height, width = logits.shape
        probability = torch.softmax(logits.reshape(batch, -1).float(), dim=-1)
        rows = torch.arange(height, device=logits.device, dtype=probability.dtype)
        cols = torch.arange(width, device=logits.device, dtype=probability.dtype)
        grid_rows = rows.view(1, height, 1).expand(batch, height, width).reshape(batch, -1)
        grid_cols = cols.view(1, 1, width).expand(batch, height, width).reshape(batch, -1)
        row = (probability * grid_rows).sum(dim=-1)
        col = (probability * grid_cols).sum(dim=-1)
        return torch.stack([row, col], dim=-1)

    def location_loss(
        self,
        logits: Any,
        target_xy: Any,
        target_visible: Any,
        *,
        mode: str,
        sigma: float,
    ) -> tuple[Any, Any, Any]:
        """Return ``(loss, pixel_error_sum, valid_count)``.

        ``mode="huber"`` is a pixel-unit loss on the expected position;
        ``mode="heatmap_ce"`` is a cross-entropy against a Gaussian heatmap whose
        sigma is in pixels.  Both read the same logits, so the two settings only
        change the objective, never the readout.
        """
        visible = target_visible.to(logits.device).bool()
        valid_count = visible.sum()
        if not bool(visible.any()):
            zero = logits.sum() * 0.0
            return zero, zero, valid_count
        # A window result is [B,T,H,W]; a single-frame probe is [B,H,W].  The loss
        # always works on the per-frame (B,H,W) grid, with the visibility mask
        # applied over the flattened (B*T) axis.
        target_full = target_xy.to(logits.device)
        if logits.ndim == 4:
            batch, length, height, width = logits.shape
            logits = logits.reshape(batch * length, height, width)
            # A two-dimensional visibility mask [B,T] describes the same window and
            # is flattened the same way; a one-dimensional mask is already flat.
            visible = visible.reshape(batch * length) if visible.ndim == 2 else visible
            target_full = (
                target_full.reshape(batch * length, 2) if target_full.ndim == 3 else target_full
            )
        elif logits.ndim == 3:
            _, height, width = logits.shape
        else:  # pragma: no cover - defensive
            raise ValueError(f"unexpected logits rank {logits.ndim} in the location loss")
        logits = logits[visible]
        target = target_full[visible].to(logits.dtype)
        if mode == "huber":
            prediction = self.soft_argmax(logits)
            centre = torch.stack(
                [
                    target[:, 0].clamp(0.0, 1.0) * (height - 1),
                    target[:, 1].clamp(0.0, 1.0) * (width - 1),
                ],
                dim=-1,
            )
            difference = (prediction - centre).abs()
            loss = F.huber_loss(difference, torch.zeros_like(difference), reduction="mean")
            error = difference.mean(dim=-1).sum()
        elif mode == "heatmap_ce":
            heat = self.heatmap_target(target, (height, width), sigma, dtype=logits.dtype)
            log_probability = F.log_softmax(logits.reshape(logits.shape[0], -1).float(), dim=-1)
            loss = -(heat.reshape(heat.shape[0], -1) * log_probability).sum(dim=-1).mean()
            error = (
                self.soft_argmax(logits)
                .sub(
                    torch.stack(
                        [
                            target[:, 0].clamp(0.0, 1.0) * (height - 1),
                            target[:, 1].clamp(0.0, 1.0) * (width - 1),
                        ],
                        dim=-1,
                    )
                )
                .abs()
                .mean(dim=-1)
                .sum()
            )
        else:  # pragma: no cover - argparse restricts the choice
            raise ValueError(f"unknown location loss mode {mode!r}")
        return loss, error, valid_count


class MeasurementPredictionHead(nn.Module):
    """Auxiliary head predicting *held-out* measurement pixels (step 4).

    The target is the true clean value of the held-out block of frame ``t-1``.
    Those pixels are removed from the encoder input (replaced by a neighbourhood
    mean and marked invalid), so the input pixels can never bypass-copy the
    target; the explicit list of excluded pixels is written to ``run.json``.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 32, kernel: int = 3):
        super().__init__()
        padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel, padding=padding),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1),
        )

    def forward(self, features: Any) -> Any:
        return self.net(features)


class N01ProbeHeads(nn.Module):
    """Localisation + optional measurement prediction, built lazily per encoder.

    ``build_probe_heads`` inspects the encoder's first output so the heads adapt
    to whatever feature width the candidate exposes; the rest of the training
    loop only ever sees this object.
    """

    def __init__(self, config: Any, image_size: tuple[int, int], location_hidden: int = 32):
        super().__init__()
        self.config = config
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.location_hidden = int(location_hidden)
        self.location: LocationProbeHead | None = None
        self.measurement: MeasurementPredictionHead | None = None
        self.feature_channels: int = 0

    @property
    def aux_enabled(self) -> bool:
        return self.measurement is not None

    def build(self, feature_channels: int, channels: int) -> None:
        if self.location is not None:
            if feature_channels != self.feature_channels:
                raise ValueError(
                    f"probe head was built for {self.feature_channels} feature channels, "
                    f"got {feature_channels}"
                )
            return
        self.feature_channels = int(feature_channels)
        self.location = LocationProbeHead(self.feature_channels, hidden=self.location_hidden)
        if bool(getattr(self.config, "aux_measurement_prediction", True)):
            self.measurement = MeasurementPredictionHead(
                self.feature_channels, int(channels), hidden=self.location_hidden
            )

    def forward(self, features: Any) -> tuple[Any, Any | None]:
        if self.location is None:  # pragma: no cover - build() always runs first
            raise RuntimeError("probe heads were not built yet")
        logits = self.location(features, size=self.image_size)
        measurement = None if self.measurement is None else self.measurement(features)
        return logits, measurement


def to_feature_map(representation: Any) -> Any:
    """Coerce an encoder representation into ``[B, C, H, W]`` features."""
    if not isinstance(representation, torch.Tensor):  # pragma: no cover - contract drift
        raise TypeError(
            f"EncoderOutput.representation must be a tensor, got {type(representation)!r}"
        )
    if representation.ndim == 4:
        return representation
    if representation.ndim == 3:  # [B, L, D] token layout
        batch, length, width = representation.shape
        side = int(round(math.sqrt(length)))
        if side * side != length:  # pragma: no cover - defensive
            raise ValueError(f"cannot reshape {length} tokens into a square grid")
        return representation.reshape(batch, side, side, width).permute(0, 3, 1, 2)
    if representation.ndim == 2:  # [B, D]
        return representation.unsqueeze(-1).unsqueeze(-1)
    raise ValueError(f"unsupported representation shape {tuple(representation.shape)}")


def build_probe_heads(
    encoder: Any,
    image_size: tuple[int, int],
    *,
    channels: int,
    device: Any = "cpu",
    location_hidden: int = 32,
) -> N01ProbeHeads:
    """Create the shared heads and size them from a real encoder output."""
    heads = N01ProbeHeads(encoder.config, image_size, location_hidden=location_hidden).to(device)
    frame = MeasurementFrame(
        values=torch.zeros((1, channels) + tuple(image_size), dtype=torch.float32, device=device),
        valid_mask=torch.ones((1, channels) + tuple(image_size), dtype=torch.bool, device=device),
        camera_id="camera",
        time_unit="frame",
        new_measurement=torch.ones((1,), dtype=torch.bool, device=device),
        metadata={"variance": torch.zeros((1, 1, channels) + tuple(image_size), device=device)},
    )
    state = encoder.init_state(1, {"camera_ids": ["camera"], "shape": tuple(image_size)}, device)
    with torch.no_grad():
        output = encoder.step(frame, None, state, ComputeBudget(max_updates=1, max_history=1))
    features = to_feature_map(output.representation)
    heads.build(int(features.shape[1]), channels)
    return heads


def batch_times(timestamps: Any, batch_size: int, length: int, device: Any) -> Any:
    """Timestamps as ``[B,T]``, accepting the shared ``[T]`` window vector too.

    ``SequenceBatch`` documents ``[B,T]``; the synthetic task currently emits one
    window time vector ``[T]``, which carries the same meaning for every episode.
    Both are accepted so the training loop does not depend on which one arrives.
    """
    tensor = torch.as_tensor(timestamps, device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).expand(batch_size, -1)
    elif tensor.ndim == 2 and tensor.shape[0] != batch_size:
        raise ValueError(
            f"timestamps batch {tensor.shape[0]} does not match the frame batch {batch_size}"
        )
    if tensor.shape[1] != length:
        raise ValueError(f"timestamps length {tensor.shape[1]} does not match the window {length}")
    return tensor


def frame_variance_metadata(variance: Any) -> Any:
    """Shape the declared per-pixel variance for ``N01State.push``.

    ``state.push`` reads ``MeasurementFrame.metadata["variance"]``; a 4-D tensor is
    interpreted as per-batch scalars and broadcast, which breaks for ``batch > 1``
    and ``max_history > 1``.  Passing ``[B,1,C,H,W]`` makes it take the
    ``expand_as`` path instead, preserving the real per-pixel variance.  This is a
    caller-side workaround for that reader and changes no numbers.
    """
    if variance is None:
        return None
    if variance.ndim == 4:
        return variance.unsqueeze(1)
    return variance


def probe_parameter_count(heads: N01ProbeHeads) -> int:
    return int(
        sum(parameter.numel() for parameter in heads.parameters() if parameter.requires_grad)
    )


def batch_parameter_count(encoder: Any) -> int:
    return int(
        sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)
    )


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------
@dataclass
class SyntheticSource:
    """Online synthetic weak-edge task plus its fixed validation task."""

    config: Any
    train_seed: int
    val_seed: int
    image_size: int = 32
    snr: str = "low"
    motion: str = "slow"
    tolerance: str = "wide"
    augment: str = "none"
    train_task: Any = None
    val_tasks: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        synthetic = _require("owac.frameworks.n01.synthetic", "--synthetic sampling")
        if self.train_task is None:
            self.train_task = synthetic.SyntheticWeakEdgeTask(
                self.config,
                image_size=self.image_size,
                snr=self.snr,
                motion=self.motion,
                tolerance=self.tolerance,
                seed=self.train_seed,
                augment=self.augment,
            )
        if not self.val_tasks:
            # The validation pool deliberately contains (speed, contrast,
            # tolerance, event) combinations that the default training task never
            # samples: still/slow/high-contrast/wide vs fast/high-contrast/narrow.
            pool = (
                ("fast", "high", "narrow", "none"),
                ("still", "high", "narrow", "none"),
                ("slow", "high", "wide", "none"),
                ("fast", "low", "narrow", "none"),
            )
            for index, (motion, snr, tolerance, augment) in enumerate(pool):
                self.val_tasks[f"val_{index}"] = synthetic.SyntheticWeakEdgeTask(
                    self.config,
                    image_size=self.image_size,
                    snr=snr,
                    motion=motion,
                    tolerance=tolerance,
                    seed=self.val_seed + index,
                    augment=augment,
                )

    def train_batch(self, batch_size: int, length: int) -> Any:
        index = getattr(self, "sample_index", 0)
        self.sample_index = index + 1
        return self.train_task.sample(batch_size, length, seed=self.train_seed + index)

    def calibration_batch(self, batch_size: int, length: int) -> Any:
        return self.train_task.sample(batch_size, length, seed=self.train_seed + 500_000)

    def validation_combinations(self) -> list[str]:
        return list(self.val_tasks)

    def val_batch(self, name: str, batch_size: int, length: int) -> Any:
        return self.val_tasks[name].sample(batch_size, length)

    def spec(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "synthetic",
            "image_size": self.image_size,
            "train_seed": self.train_seed,
            "val_seed": self.val_seed,
            "train_spec": self.train_task.spec(),
            "validation_combinations": {name: task.spec() for name, task in self.val_tasks.items()},
        }
        return payload


# ---------------------------------------------------------------------------
# loss weights and sequence stepping
# ---------------------------------------------------------------------------
def loss_weights_from_args(config: Any, raw: str | None) -> dict[str, float]:
    """Resolve the three loss weights: config defaults, then CLI overrides."""
    weights = {str(key): float(value) for key, value in dict(config.loss_weights).items()}
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise SystemExit(f"--loss-weights entries must look like name=value, got {item!r}")
            name, value = item.split("=", 1)
            name = name.strip()
            if name not in weights:
                raise SystemExit(
                    f"unknown loss weight {name!r}; known weights are {sorted(weights)}"
                )
            weights[name] = float(value)
    for name in ("location", "measurement_prediction", "budget"):
        if name not in weights:
            raise SystemExit(f"loss weight {name!r} is missing")
    return weights


@dataclass
class SequenceStepResult:
    """Everything one online sequence pass measured, with live graphs kept."""

    logits: Any
    measurement_prediction: Any | None
    target: Any
    target_visible: Any
    target_xy: Any
    features: list[Any]
    state: Any
    step_times_ms: list[float]
    updates: int
    last_output: EncoderOutput | None
    last_frame: Any = None
    measurement_target: Any | None = None
    measurement_weight: Any | None = None
    measurement_valid_count: int = 0
    held_out_frames: int = 0
    sample_age: Any | None = None


def _split_sequence(batch: Any) -> tuple[Any, Any, Any, Any]:
    values = batch.values
    valid = batch.valid
    variance = batch.variance
    edge_target = batch.edge_target
    return values, valid, variance, edge_target


def sequence_step(
    encoder: Any,
    heads: N01ProbeHeads,
    batch: Any,
    *,
    step: int,
    state: Any = None,
    tbptt: int = 4,
    budget: ComputeBudget | None = None,
    grad_enabled: bool = True,
    reset_at_start: bool = True,
    device: Any = "cpu",
    holdout: bool = True,
) -> SequenceStepResult:
    """Run one full causal window through ``N01Encoder.step`` frame by frame.

    ``reset_at_start`` performs the plain-random-batch reset at the window start;
    ``tbptt`` detaches the state every N frames.  The same function is used for
    training (``grad_enabled=True``) and for validation/inference.
    """
    values, valid, variance, _ = _split_sequence(batch)
    batch_size, length = int(values.shape[0]), int(values.shape[1])
    size = (int(values.shape[-2]), int(values.shape[-1]))
    times = batch_times(batch.timestamps, batch_size, length, values.device)
    if state is None:
        state = encoder.init_state(
            batch_size, {"camera_ids": ["camera"], "shape": size}, values.device
        )
    elif reset_at_start:
        state = encoder.reset(state, torch.ones(batch_size, dtype=torch.bool, device=values.device))

    # The holdout offsets for this step are applied per frame by holdout_mask().
    budget = budget or ComputeBudget(
        max_updates=encoder.config.num_blocks, max_history=encoder.config.max_history
    )
    context = torch.enable_grad() if grad_enabled else torch.no_grad()
    logits: list[Any] = []
    measurement_predictions: list[Any] = []
    targets: list[Any] = []
    visible: list[Any] = []
    features: list[Any] = []
    timings: list[float] = []
    measurement_targets: list[Any] = []
    measurement_valid = 0
    measurement_weights: list[Any] = []
    last_output: EncoderOutput | None = None
    updates = 0
    sample_ages: list[Any] = []

    with context:
        for index in range(length):
            frame_values = values[:, index].to(device)
            frame_valid = valid[:, index].to(device).bool()
            frame_variance = variance[:, index].to(device)
            if holdout:
                # Hold out the target block from the input: replace it by a local
                # neighbourhood mean and mark it invalid, so no input pixel is its
                # own measurement-prediction target.  ``holdout=False`` leaves the
                # measurements untouched, which is what an online/replay path does
                # and what the frame-by-frame consistency check needs.
                mask = holdout_mask(size[0], size[1], step, device, frame_values.dtype)
                frame_values = frame_values.masked_fill(mask.bool().view(1, 1, *size), 0.0)
                frame_valid = frame_valid & ~mask.bool().view(1, 1, *size)
            frame = MeasurementFrame(
                values=frame_values,
                valid_mask=frame_valid,
                camera_id=str(batch.meta.get("camera_id", "camera")),
                timestamp=times[:, index].to(device),
                time_unit="frame",
                new_measurement=batch.new_measurement[:, index].to(device).bool(),
                metadata={"variance": frame_variance_metadata(frame_variance)},
            )
            started = time.perf_counter()
            output = encoder.step(frame, None, state, budget)
            timings.append((time.perf_counter() - started) * 1000.0)
            state = output.state
            # Algorithmic lag (how old the measurements behind this estimate are),
            # kept separate from the wall-clock processing time in ``timings``.
            sample_ages.append(
                torch.full(
                    (batch_size,), output.diagnostics.get("mean_age", 0.0), device=values.device
                )
            )
            feature = to_feature_map(output.representation)
            heads.build(int(feature.shape[1]), int(values.shape[2]))
            frame_logits, frame_measurement = heads(feature)
            logits.append(frame_logits)
            features.append(feature)
            targets.append(batch.edge_target[:, index].to(device))
            visible.append(batch.target_visible[:, index].to(device))
            if index >= 1:
                mask = holdout_mask(size[0], size[1], step, device, frame_values.dtype).bool()
                # A repeated/padding target frame carries no new independent
                # evidence (``new_measurement`` is False and its frame id repeats),
                # so its pixels are weighted out of the auxiliary target instead of
                # being counted twice.  The holdout mask is two-dimensional, so it
                # indexes the (H, W) axes of both the prediction and the target;
                # the held-out pixel count is the same at every step, which keeps
                # the per-step tensors stackable.
                fresh = batch.new_measurement[:, index].to(device).bool().view(batch_size, 1, 1)
                source_valid = valid[:, index - 1].to(device).bool()[:, :, mask]
                weight = source_valid & fresh
                prediction = frame_measurement[:, :, mask]
                target = values[:, index - 1].to(device)[:, :, mask]
                measurement_predictions.append(prediction)
                measurement_targets.append(target)
                measurement_weights.append(weight)
                measurement_valid += int(weight.sum())
            else:
                measurement_predictions.append(None)
            last_output = output
            updates += int(output.cost.get("updates", 0) or 0) if output.cost else 0
            if tbptt > 0 and (index + 1) % tbptt == 0 and index + 1 < length:
                state = encoder.detach_state(state)
            # A frame that is not a new measurement carries no new evidence; the
            # encoder already deduplicates it, and the batch window start resets.

    # The held-out block has the same pixel count at every step, so the auxiliary
    # targets stack into [B, T-1, C, P]; invalid or repeated frames stay in the
    # tensor but are weighted out by ``measurement_weight``.
    selected = [item for item in measurement_predictions[1:] if item is not None]
    if selected:
        prediction_stack = torch.stack(selected, dim=1)
        target_stack = torch.stack(measurement_targets, dim=1)
        weight_stack = torch.stack(measurement_weights, dim=1)
    else:
        prediction_stack = None
        target_stack = None
        weight_stack = None
    return SequenceStepResult(
        logits=torch.stack(logits, dim=1),
        measurement_prediction=prediction_stack,
        target=torch.stack(targets, dim=1),
        target_visible=torch.stack(visible, dim=1),
        target_xy=batch.target_xy.to(device),
        features=features,
        state=state,
        step_times_ms=timings,
        updates=updates,
        last_output=last_output,
        measurement_target=target_stack,
        measurement_weight=weight_stack,
        measurement_valid_count=measurement_valid,
        held_out_frames=length - 1,
        sample_age=torch.stack(sample_ages, dim=1) if sample_ages else None,
    )


@dataclass
class LossBreakdown:
    total: Any
    location: Any
    location_error_px: float
    location_valid_pixels: int
    measurement: Any
    measurement_valid_pixels: int
    budget: Any
    updates: int
    weights: dict[str, float] = field(default_factory=dict)

    def as_dict(self, step: int) -> dict[str, Any]:
        return {
            "step": int(step),
            "loss": float(self.total.detach()),
            "loss_location": float(self.location.detach()),
            "loss_measurement": float(self.measurement.detach()),
            "loss_budget": float(self.budget.detach()),
            "valid_pixels": int(self.location_valid_pixels),
            "valid_measurement_pixels": int(self.measurement_valid_pixels),
            "location_error_px": float(self.location_error_px),
            "updates": int(self.updates),
        }

    def log_line(self, step: int, wall_ms: float) -> str:
        return json.dumps(
            {
                "step": int(step),
                "loss": round(float(self.total.detach()), 6),
                "loss_location": round(float(self.location.detach()), 6),
                "loss_measurement": round(float(self.measurement.detach()), 6),
                "loss_budget": round(float(self.budget.detach()), 6),
                "valid_pixels": int(self.location_valid_pixels),
                "valid_measurement_pixels": int(self.measurement_valid_pixels),
                "location_error_px": round(float(self.location_error_px), 4),
                "updates": int(self.updates),
                "step_ms": round(float(wall_ms), 2),
            },
            sort_keys=True,
        )


def train_one_step(
    encoder: Any,
    heads: N01ProbeHeads,
    batch: Any,
    optimizer: Any,
    *,
    step: int,
    weights: dict[str, float],
    location_mode: str = "huber",
    sigma: float = 1.5,
    target_updates: float = _DEFAULT_TARGET_UPDATES,
    tbptt: int = 4,
    grad_clip: float | None = 1.0,
    device: Any = "cpu",
) -> LossBreakdown:
    """One optimiser step: full causal window, then a single backward pass."""
    optimizer.zero_grad(set_to_none=True)
    result = sequence_step(
        encoder,
        heads,
        batch,
        step=step,
        tbptt=tbptt,
        budget=ComputeBudget(max_updates=1, max_history=int(encoder.config.max_history)),
        grad_enabled=True,
        device=device,
    )
    location_loss, location_error, location_valid = heads.location.location_loss(
        result.logits, result.target_xy, result.target_visible, mode=location_mode, sigma=sigma
    )
    if result.measurement_prediction is not None and weights["measurement_prediction"] != 0.0:
        prediction = result.measurement_prediction
        target = result.measurement_target
        # Held-out pixels that are duplicated, padding or invalid stay in the
        # tensor but carry zero weight, so they never contribute to the loss.
        valid_target = torch.isfinite(target) & torch.isfinite(prediction)
        if result.measurement_weight is not None:
            valid_target = valid_target & result.measurement_weight.to(valid_target.device)
        denominator = valid_target.sum().clamp_min(1)
        difference = (prediction - torch.nan_to_num(target, nan=0.0)).abs()
        measurement_loss = (difference * valid_target).sum() / denominator
        measurement_valid = int(valid_target.sum())
    else:
        measurement_loss = location_loss.detach() * 0.0
        measurement_valid = 0
    if weights["budget"] != 0.0:
        # Effective update count actually performed by the encoder this window,
        # compared against the declared target.  The task error stays in the same
        # log line so a low cost value can never hide a task regression.
        effective_updates = float(result.updates if result.updates else result.held_out_frames)
        budget_loss = torch.as_tensor(
            (effective_updates - float(target_updates)) ** 2,
            dtype=location_loss.dtype,
            device=location_loss.device,
        )
    else:
        budget_loss = torch.zeros((), dtype=location_loss.dtype, device=location_loss.device)
    total = (
        weights["location"] * location_loss
        + weights["measurement_prediction"] * measurement_loss
        + weights["budget"] * budget_loss
    )
    total.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in list(encoder.parameters()) + list(heads.parameters()) if p.requires_grad],
            float(grad_clip),
        )
    optimizer.step()
    return LossBreakdown(
        total=total,
        location=location_loss,
        location_error_px=float(location_error.detach()) / max(int(location_valid), 1),
        location_valid_pixels=int(location_valid),
        measurement=measurement_loss,
        measurement_valid_pixels=measurement_valid,
        budget=budget_loss,
        updates=int(result.updates),
        weights=dict(weights),
    )


@dataclass
class ValidationResult:
    loss: float
    location_loss: float
    measurement_loss: float
    location_error_px: float
    valid_pixels: int
    valid_measurement_pixels: int
    batches: int
    combination: str


def evaluate_sequence(
    encoder: Any,
    heads: N01ProbeHeads,
    batch: Any,
    *,
    step: int,
    weights: dict[str, float],
    location_mode: str = "huber",
    sigma: float = 1.5,
    device: Any = "cpu",
) -> ValidationResult:
    """Validation on a fixed independent seed, without gradients or updates."""
    with torch.no_grad():
        result = sequence_step(
            encoder,
            heads,
            batch,
            step=step,
            tbptt=0,
            budget=ComputeBudget(max_updates=1, max_history=int(encoder.config.max_history)),
            grad_enabled=False,
            device=device,
        )
        location_loss, location_error, location_valid = heads.location.location_loss(
            result.logits,
            result.target_xy,
            result.target_visible,
            mode=location_mode,
            sigma=sigma,
        )
        if result.measurement_prediction is not None:
            target = result.measurement_target
            valid_target = torch.isfinite(target) & torch.isfinite(result.measurement_prediction)
            denominator = valid_target.sum().clamp_min(1)
            difference = (result.measurement_prediction - torch.nan_to_num(target, nan=0.0)).abs()
            measurement_loss = float(((difference * valid_target).sum() / denominator).detach())
            measurement_valid = int(valid_target.sum())
        else:
            measurement_loss = float("nan")
            measurement_valid = 0
    return ValidationResult(
        loss=float(
            (
                weights["location"] * location_loss
                + weights["measurement_prediction"] * measurement_loss
            ).detach()
        )
        if measurement_loss == measurement_loss
        else float(location_loss.detach()),
        location_loss=float(location_loss.detach()),
        measurement_loss=measurement_loss,
        location_error_px=float(location_error.detach()) / max(int(location_valid), 1),
        valid_pixels=int(location_valid),
        valid_measurement_pixels=measurement_valid,
        batches=1,
        combination="",
    )


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------
def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    numpy_state = np.random.get_state()
    state["numpy"] = [
        numpy_state[0],
        numpy_state[1].tolist(),
        int(numpy_state[2]),
        *numpy_state[3:],
    ]
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "python" in state:
        random.setstate(
            tuple(state["python"]) if isinstance(state["python"], list) else state["python"]
        )
    if "numpy" in state:
        kind, keys, position, has_gauss, cached = state["numpy"]
        np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


def checkpoint_state(
    encoder: Any,
    heads: N01ProbeHeads,
    optimizer: Any,
    *,
    step: int,
    seed: int,
    history: Sequence[dict[str, Any]],
    config_payload: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """The frozen ``checkpoint.pt`` payload."""
    return {
        "config": dict(config_payload),
        "model": encoder.state_dict(),
        "probe": heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "seed": int(seed),
        "history": [dict(row) for row in history],
        "git_sha": git_sha(repo_root),
        "torch_version": str(torch.__version__),
        "rng_state": rng_state(),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: Any = "cpu") -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise ValueError(f"checkpoint {path} is not a dict payload")
    missing = {"config", "model", "step", "seed"} - set(payload)
    if missing:
        raise ValueError(f"checkpoint {path} is missing fields {sorted(missing)}")
    return payload


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainingSettings:
    seed: int = 0
    steps: int = 200
    batch_size: int = 8
    length: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    tbptt: int = 4
    log_every: int = 1
    val_every: int = 0
    val_batches: int = 1
    location_mode: str = "huber"
    sigma: float = 1.5
    target_updates: float = _DEFAULT_TARGET_UPDATES
    device: str = "cpu"
    image_size: int = 32
    losses: dict[str, float] = field(default_factory=dict)


def train(
    encoder: Any,
    heads: N01ProbeHeads,
    source: SyntheticSource,
    settings: TrainingSettings,
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
    repo_root: Path,
    resume: dict[str, Any] | None = None,
    log_stream: Any = None,
) -> dict[str, Any]:
    """Run the online training loop and return the run record."""
    log_stream = log_stream or sys.stdout
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed % (2**32))
    random.seed(settings.seed)
    encoder.train()
    heads.train()
    parameters = [p for p in encoder.parameters() if p.requires_grad] + [
        p for p in heads.parameters() if p.requires_grad
    ]
    if not parameters:
        raise SystemExit("no trainable parameters: the encoder and probe head are fully frozen")
    optimizer = torch.optim.AdamW(parameters, lr=settings.lr, weight_decay=settings.weight_decay)
    history: list[dict[str, Any]] = []
    start_step = 0
    validation_log: list[dict[str, Any]] = []
    if resume is not None:
        encoder.load_state_dict(resume["model"])
        if "probe" in resume:
            heads.load_state_dict(resume["probe"])
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        history = [dict(row) for row in resume.get("history", [])]
        start_step = int(resume.get("step", 0))
        restore_rng_state(resume.get("rng_state", {}))
        source.sample_index = int(resume.get("source_sample_index", start_step))
        print(
            json.dumps(
                {
                    "resume": True,
                    "from_step": start_step,
                    "history_rows": len(history),
                    "git_sha": resume.get("git_sha"),
                },
                sort_keys=True,
            ),
            file=log_stream,
            flush=True,
        )
    if int(encoder.calibration_count) == 0:
        encoder.calibrating = True
        sequence_step(
            encoder,
            heads,
            source.calibration_batch(settings.batch_size, settings.length),
            step=0,
            tbptt=0,
            grad_enabled=False,
            device=settings.device,
            holdout=False,
        )
        encoder.calibrating = False
    step_times: list[float] = []
    baseline_rss = peak_rss_bytes()
    started = time.perf_counter()
    for offset in range(settings.steps):
        step = start_step + offset + 1
        batch = source.train_batch(settings.batch_size, settings.length)
        tick = time.perf_counter()
        breakdown = train_one_step(
            encoder,
            heads,
            batch,
            optimizer,
            step=step,
            weights=settings.losses,
            location_mode=settings.location_mode,
            sigma=settings.sigma,
            target_updates=settings.target_updates,
            tbptt=settings.tbptt,
            grad_clip=settings.grad_clip,
            device=settings.device,
        )
        wall_ms = (time.perf_counter() - tick) * 1000.0
        step_times.append(wall_ms)
        row = breakdown.as_dict(step)
        row["wall_ms"] = wall_ms
        history.append(row)
        if settings.log_every > 0 and (
            step % settings.log_every == 0 or step == start_step + settings.steps
        ):
            print(breakdown.log_line(step, wall_ms), file=log_stream, flush=True)
        if settings.val_every > 0 and step % settings.val_every == 0:
            names = source.validation_combinations()
            name = names[(step // settings.val_every) % len(names)]
            validation = evaluate_sequence(
                encoder,
                heads,
                source.val_batch(name, settings.batch_size, settings.length),
                step=step,
                weights=settings.losses,
                location_mode=settings.location_mode,
                sigma=settings.sigma,
                device=settings.device,
            )
            encoder.train()
            heads.train()
            validation_log.append(
                {
                    "step": step,
                    "combination": name,
                    "loss": validation.loss,
                    "loss_location": validation.location_loss,
                    "loss_measurement": validation.measurement_loss,
                    "location_error_px": validation.location_error_px,
                    "valid_pixels": validation.valid_pixels,
                    "valid_measurement_pixels": validation.valid_measurement_pixels,
                }
            )
            print(
                json.dumps({"validation": validation_log[-1]}, sort_keys=True),
                file=log_stream,
                flush=True,
            )
    elapsed = time.perf_counter() - started
    checkpoint_path = output_dir / "checkpoint.pt"
    payload = checkpoint_state(
        encoder,
        heads,
        optimizer,
        step=start_step + settings.steps,
        seed=settings.seed,
        history=history,
        config_payload={**config_payload, "loss_mode": settings.location_mode},
        repo_root=repo_root,
    )
    payload["source_sample_index"] = getattr(source, "sample_index", 0)
    save_checkpoint(checkpoint_path, payload)
    sorted_times = sorted(step_times) or [0.0]
    run: dict[str, Any] = {
        "kind": "training",
        "config": config_payload,
        "settings": {
            "seed": settings.seed,
            "steps": settings.steps,
            "batch_size": settings.batch_size,
            "length": settings.length,
            "lr": settings.lr,
            "weight_decay": settings.weight_decay,
            "grad_clip": settings.grad_clip,
            "tbptt": settings.tbptt,
            "losses": settings.losses,
            "location_mode": settings.location_mode,
            "location_sigma_px": settings.sigma,
            "target_updates": settings.target_updates,
            "aux_measurement_prediction": bool(heads.aux_enabled),
            "device": settings.device,
        },
        "history": history,
        "validation": validation_log,
        "checkpoint": str(checkpoint_path),
        "checkpoint_fields": sorted(payload),
        "parameters": {
            "encoder_trainable": batch_parameter_count(encoder),
            "probe_trainable": probe_parameter_count(heads),
        },
        "resources": {
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_baseline_bytes": baseline_rss,
            "peak_gpu_bytes": peak_gpu_bytes(),
            "gpu_available": bool(torch.cuda.is_available()),
        },
        "timing_ms": {
            "step_p50": statistics.median(sorted_times),
            "step_p95": _percentile(sorted_times, 0.95),
            "step_mean": float(sum(sorted_times) / len(sorted_times)),
            "total_s": elapsed,
        },
        "data_spec": source.spec(),
        "measurement_prediction_exclusion": {
            "holdout_period": _HOLDOUT_PERIOD,
            "holdout_offsets": list(_HOLDOUT_OFFSETS),
            "rule": (
                "pixels of frame t-1 whose (row%2,col%2) equals the step's offset are the "
                "prediction target; the same pixels are replaced by a 3x3 neighbourhood mean "
                "and marked invalid in every encoder input frame, so no input pixel is its own "
                "target and the target value never enters the encoder"
            ),
        },
        "units": units_note(),
        "git_sha": payload["git_sha"],
        "torch_version": payload["torch_version"],
        "command": " ".join([Path(sys.executable).name, "-m", "owac.training.n01", *sys.argv[1:]]),
        "argv": list(sys.argv),
    }
    write_json(output_dir / "run.json", run)
    return run


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(quantile * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.training.n01",
        description="Train the N01 adaptive spatio-temporal evidence integrator (CPU, FP32).",
    )
    parser.add_argument(
        "--config", required=True, help="N01 config JSON (configs/encoders/n01/*.json)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="online synthetic weak-edge task")
    source.add_argument("--npz-root", type=Path, help="real RawVLA NPZ dataset root")
    parser.add_argument("--npz-config", type=Path, help="RawVLA dataset config JSON for --npz-root")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="explicit training seed")
    parser.add_argument("--val-seed", type=int, default=None, help="fixed validation seed")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--length", type=int, default=8, help="causal window length in frames")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--tbptt", type=int, default=4, help="detach the state every N frames")
    parser.add_argument(
        "--log-every", type=int, default=1, help="print one JSON line every N steps"
    )
    parser.add_argument("--val-every", type=int, default=0, help="run validation every N steps")
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument(
        "--loss",
        choices=("huber", "heatmap_ce"),
        default="huber",
        help="location objective: pixel-unit Huber, or heatmap cross-entropy",
    )
    parser.add_argument("--sigma", type=float, default=1.5, help="heatmap sigma in pixels")
    parser.add_argument(
        "--loss-weights",
        default=None,
        help="override, e.g. budget=0.001,measurement_prediction=0.0",
    )
    parser.add_argument("--target-updates", type=float, default=_DEFAULT_TARGET_UPDATES)
    parser.add_argument("--no-aux", action="store_true", help="disable MeasurementPredictionHead")
    parser.add_argument("--image-size", type=int, default=32, help="synthetic measurement size")
    parser.add_argument("--snr", default="low")
    parser.add_argument("--motion", default="slow", choices=("still", "slow", "fast"))
    parser.add_argument("--tolerance", default="wide", choices=("wide", "narrow"))
    parser.add_argument(
        "--augment",
        default="none",
        choices=("none", "sudden_appear", "direction_reversal", "occlusion_release", "crossing"),
    )
    parser.add_argument(
        "--device", default="cpu", help="CPU/FP32 by default, as the protocol fixes"
    )
    parser.add_argument(
        "--weights", type=Path, help="pinned pi0.5 safetensors for real NPZ training"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="pinned PaliGemma tokenizer for NPZ training"
    )
    parser.add_argument("--suite", choices=("libero", "robotwin2"), default="libero")
    parser.add_argument("--resume", type=Path, default=None, help="continue from checkpoint.pt")
    return parser


def _npz_source(args: argparse.Namespace, config: Any) -> Any:
    module = _require(
        "owac.raw.rawvla_sequence",
        "--npz-root training",
    )
    if not hasattr(module, "RawVLASequenceDataset"):
        raise _OptionalDependencyError(
            "owac.raw.rawvla_sequence exists but exposes no RawVLASequenceDataset; "
            "the real NPZ loader is still being implemented in parallel."
        )
    if args.npz_config is None:
        raise SystemExit("--npz-root requires --npz-config pointing at the dataset config JSON")
    if not Path(args.npz_config).is_file():
        raise SystemExit(f"NPZ config not found: {args.npz_config}")
    payload = json.loads(Path(args.npz_config).read_text())
    return module.RawVLASequenceDataset(root=args.npz_root, **payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if not args.synthetic:
        if args.weights is None or args.tokenizer is None or args.npz_config is None:
            raise SystemExit(
                "real NPZ training requires --weights, --tokenizer and --npz-config; "
                "it uses native pi0.5 actions, not a synthetic location target"
            )
        action = _require("owac.training.n01_action", "real NPZ native-action training")
        command = [
            "--config",
            args.config,
            "--root",
            str(args.npz_root),
            "--suite",
            args.suite,
            "--weights",
            str(args.weights),
            "--tokenizer",
            str(args.tokenizer),
            "--dataset-config",
            str(args.npz_config),
            "--output",
            str(args.output),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--window",
            str(args.length),
            "--device",
            args.device,
            "--lr",
            str(args.lr),
        ]
        if args.resume:
            command.extend(["--resume", str(args.resume)])
        return action.main(command)
    config_module = _require("owac.frameworks.n01.config", "config loading")
    model_module = _require("owac.frameworks.n01.model", "the N01 encoder")
    config = config_module.load_config(args.config)
    fingerprint = config_module.config_fingerprint(config)
    if args.no_aux:
        config = config.replace(aux_measurement_prediction=False)
    weights = loss_weights_from_args(config, args.loss_weights)
    config_payload = {
        **config.to_dict(),
        "config_path": str(args.config),
        "config_fingerprint": fingerprint,
        "loss_weights": weights,
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    torch.manual_seed(args.seed)
    encoder = model_module.N01Encoder(config).to(args.device)
    heads = build_probe_heads(
        encoder,
        (args.image_size, args.image_size),
        channels=int(config.channels),
        device=args.device,
    )
    if args.no_aux and heads.measurement is not None:  # pragma: no cover - defensive
        raise SystemExit("--no-aux did not disable the measurement prediction head")
    if args.synthetic:
        source: Any = SyntheticSource(
            config,
            train_seed=args.seed,
            val_seed=(args.seed + 1_000_003) if args.val_seed is None else args.val_seed,
            image_size=args.image_size,
            snr=args.snr,
            motion=args.motion,
            tolerance=args.tolerance,
            augment=args.augment,
        )
    else:
        source = _npz_source(args, config)
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume is not None:
        stored = dict(resume.get("config", {}))
        stored_fingerprint = stored.get("config_fingerprint")
        if stored_fingerprint and stored_fingerprint != fingerprint:
            print(
                json.dumps(
                    {
                        "warning": "config fingerprint differs from the checkpoint",
                        "checkpoint": stored_fingerprint,
                        "current": fingerprint,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    settings = TrainingSettings(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        length=args.length,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=(None if args.grad_clip <= 0 else args.grad_clip),
        tbptt=args.tbptt,
        log_every=args.log_every,
        val_every=args.val_every,
        val_batches=args.val_batches,
        location_mode=args.loss,
        sigma=args.sigma,
        target_updates=args.target_updates,
        device=args.device,
        image_size=args.image_size,
        losses=weights,
    )
    run = train(
        encoder,
        heads,
        source,
        settings,
        output_dir=output_dir,
        config_payload=config_payload,
        repo_root=repo_root,
        resume=resume,
    )
    print(
        json.dumps(
            {
                "run": str(output_dir / "run.json"),
                "checkpoint": run["checkpoint"],
                "final_step": int(run["history"][-1]["step"]) if run["history"] else 0,
                "timing_ms": run["timing_ms"],
                "peak_rss_bytes": run["resources"]["peak_rss_bytes"],
                "peak_gpu_bytes": run["resources"]["peak_gpu_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

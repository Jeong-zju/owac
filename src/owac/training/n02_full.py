"""P1 training entry point for the N02 full (v1.1) line.

    python -m owac.training.n02_full --stage p1-pretrain    --config ... \
        [--seed N] [--updates N] [--lr-scale S] [--output DIR] [--resume PATH]
    python -m owac.training.n02_full --stage p1-measurement --config ... [same options]

What this module owns (P1, T4)
------------------------------
* the two training loops: RAW Base masked pretraining (FCMAE, real frozen
  anchors) and the fixed-candidate measurement stage (controlled synthetic tasks
  plus real pseudo-RAW windows), with AdamW, warmup+cosine, gradient clipping and
  gradient accumulation to the declared *effective* batch;
* checkpointing that stores model/optimizer/scheduler/AMP/RNG/sampler position/
  versions/partition hash, and a resume path that reproduces the uninterrupted
  run update for update;
* the loss composition. The online path stays inside ``model.step_bundle``; this
  harness reads the raw quantities it returns (``locked_checks`` for the
  prequential score, ``fit_scores`` for the fit score, ``worlds[*].score``) and
  composes the declared terms. Supervision enters only through
  :class:`owac.frameworks.n02_full.state.TrainingTargets`.
* instrumentation: per-term activation counts, per-term per-valid-scalar mean and
  density sum (recorded separately, never mixed into one number), Base/head
  gradient norms and the angle between the two gradient vectors.

Failure policy
--------------
A missing upstream module, a missing data root or an absent checkpoint becomes
``status="not_run"`` with the exact reason and exit code 0 (the P0 CLI's policy).
Nothing is fabricated: a loss term whose explicit label or truth geometry is
absent is recorded as ``unavailable`` with its reason and excluded from the
objective; it is never written as a zero-valued success. A malformed config, an
unknown CLI value or an unexpected runtime failure exits non-zero and still
leaves a receipt.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from owac.frameworks.n02_full import config as n02_config

__all__ = [
    "CHECKPOINT_FORMAT",
    "LOSS_TERMS",
    "MEASUREMENT_DEFAULTS",
    "PRETRAIN_DEFAULTS",
    "STAGES",
    "CheckpointSpec",
    "DataSpec",
    "MeasurementSpec",
    "P1Trainer",
    "PretrainSpec",
    "TrainNotAvailable",
    "TrainingSpecError",
    "build_parser",
    "loss_terms_from_step",
    "main",
    "stack_measurement_frames",
]

#: Training stages this CLI implements.
STAGES = ("p1-pretrain", "p1-measurement")

#: Declared ablation arms of the measurement stage. ``main`` is the frozen
#: composition (1:1:0.1). The two loss-weight arms reuse the same step_bundle and
#: targets and only re-weight the terms; the input ablations and the official
#: ratio losses need interfaces this harness does not have yet and are reported
#: as not_run with the exact missing piece rather than approximated.
MEASUREMENT_ARMS = (
    "main",
    "candidate_discrimination",
    "conditional_likelihood",
    "raw_frame",
    "conditional_innovation",
    "registration",
    "nre",
    "bnre",
)

ARM_LOSS_WEIGHTS = {
    "candidate_discrimination": {
        "raw_proper_prediction": 0.0,
        "candidate_ce": 1.0,
        "geometry_aux": 0.1,
    },
    "conditional_likelihood": {
        "raw_proper_prediction": 1.0,
        "candidate_ce": 0.0,
        "geometry_aux": 0.1,
    },
}

ARM_MISSING_INTERFACES = {
    "raw_frame": (
        "needs an input-domain switch in the T1 observation encoder (score the raw frame "
        "instead of the conditional prediction)"
    ),
    "conditional_innovation": (
        "needs a T1/T2 conditional-innovation input mode (whiten against the history "
        "prediction before encoding)"
    ),
    "registration": ("needs a T2 registration/eta-warp input mode (align frames before encoding)"),
}

DEFAULT_CONFIGS = {
    "p1-pretrain": "configs/encoders/n02_full/p1_pretrain.json",
    "p1-measurement": "configs/encoders/n02_full/p1_measurement.json",
}
DEFAULT_OUTPUT_ROOT = "outputs/n02-full"

#: Checkpoint container format; a loaded checkpoint must declare it.
CHECKPOINT_FORMAT = "n02-full-p1-checkpoint-v1"
#: Run-manifest format written next to every run.
RUN_MANIFEST_FORMAT = "n02-full-p1-run-manifest-v1"

#: Loss columns of the measurement stage. ``raw_proper_prediction`` is ONE target:
#: its prequential and fit proper scores are separate columns and are never summed
#: into a second likelihood (interface freeze §3.7).
LOSS_TERMS = (
    "raw_proper_prediction_prequential",
    "raw_proper_prediction_fit",
    "candidate_ce",
    "geometry_aux",
)

#: Declared defaults, mirrored from the P1 stage prompt. They are documentation of
#: the frozen starting values; the config file is the authority at run time.
PRETRAIN_DEFAULTS = {
    "updates": 100_000,
    "effective_batch_frames": 256,
    "micro_batch_frames": 16,
    "lr": 1.5e-4,
    "lr_grid": [7.5e-5, 1.5e-4, 3e-4],
    "weight_decay": 0.05,
    "betas": [0.9, 0.95],
    "warmup_fraction": 0.05,
    "mask_ratio": 0.6,
}
MEASUREMENT_DEFAULTS = {
    "updates": 50_000,
    "effective_batch_episodes": 32,
    "frames_per_episode": 8,
    "lr_base": 3e-5,
    "lr_head": 3e-4,
    "lr_grid_scales": [0.5, 1.0, 2.0],
    "weight_decay": 0.01,
    "loss_weights": {"raw_proper_prediction": 1.0, "candidate_ce": 1.0, "geometry_aux": 0.1},
}


class TrainingSpecError(n02_config.ConfigError):
    """A resolved training section is malformed; the message names the key path."""


class TrainNotAvailable(RuntimeError):
    """A required module or asset is absent; the run is ``not_run`` with this reason."""

    def __init__(self, reason: str, *, missing: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.missing = tuple(missing)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def configure_precision(amp: str, device: str) -> dict[str, Any]:
    """Apply the declared numerical mode and report what was actually set.

    ``tf32`` keeps FP32 storage and lets the CUDA matmul/conv kernels use the TF32
    path; ``bf16`` autocasts the forward to bfloat16. Both are only usable after the
    declared error check, so the applied settings are returned for the receipt.
    """
    torch = _torch()
    payload: dict[str, Any] = {"amp": str(amp), "device": str(device)}
    if str(device).startswith("cuda"):
        if amp == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
    payload["matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    payload["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    payload["float32_matmul_precision"] = str(torch.get_float32_matmul_precision())
    return payload


def autocast_context(amp: str, device: str) -> Any:
    """The forward context for one optimisation step."""
    torch = _torch()
    if amp == "bf16" and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


def _torch() -> Any:
    import torch

    return torch


def _numpy() -> Any:
    import numpy as np

    return np


def _now() -> str:

    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write one JSON document, creating parents."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    return target


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default) + "\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON document, transparently handling a ``.gz`` suffix."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"missing JSON asset: {target}")
    if target.suffix == ".gz":
        with gzip.open(target, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(target.read_text(encoding="utf-8"))


def _derive_seed(*parts: Any) -> int:
    """Data-side seed derivation, reusing the frozen P0 generator."""
    from owac.frameworks.n02_full import synthetic as synth

    return int(synth.derive_seed(*parts))


def interpreter_info() -> dict[str, Any]:
    import platform

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for name in ("torch", "numpy"):
        try:
            module = importlib.import_module(name)
            info[name] = getattr(module, "__version__", "unknown")
            if name == "torch":
                info["cuda_available"] = bool(module.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - a receipt records the failure
            info[name] = None
            info[f"{name}_import_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() or None


# ---------------------------------------------------------------------------
# specs (strict section parsing)
# ---------------------------------------------------------------------------


def _kind_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingSpecError(f"{where}: expected an integer, got {type(value).__name__}")
    return int(value)


def _kind_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingSpecError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


def _kind_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise TrainingSpecError(f"{where}: expected a boolean, got {type(value).__name__}")
    return bool(value)


def _kind_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise TrainingSpecError(f"{where}: expected a string, got {type(value).__name__}")
    return value


def _kind_floats(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TrainingSpecError(f"{where}: expected a list of numbers, got {type(value).__name__}")
    return tuple(_kind_float(item, f"{where}[{index}]") for index, item in enumerate(value))


def _kind_strs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TrainingSpecError(f"{where}: expected a list of strings, got {type(value).__name__}")
    return tuple(_kind_str(item, f"{where}[{index}]") for index, item in enumerate(value))


def _kind_map(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise TrainingSpecError(f"{where}: expected an object, got {type(value).__name__}")
    return dict(value)


def _kind_raw(value: Any, where: str) -> Any:
    """Passthrough for a structured sub-section the owner module interprets."""
    return value


def _kind_ints(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TrainingSpecError(f"{where}: expected a list of integers")
    return tuple(_kind_int(item, f"{where}[{index}]") for index, item in enumerate(value))


_KINDS: dict[str, Any] = {
    "int": _kind_int,
    "raw": _kind_raw,
    "ints": _kind_ints,
    "float": _kind_float,
    "bool": _kind_bool,
    "str": _kind_str,
    "floats": _kind_floats,
    "strs": _kind_strs,
    "map": _kind_map,
}


def _section(cls: type, section: Any, where: str) -> Any:
    """Strictly build one frozen spec dataclass from a config section."""
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise TrainingSpecError(f"{where}: expected an object, got {type(section).__name__}")
    kinds = getattr(cls, "KINDS", None)
    if not kinds:
        raise TrainingSpecError(f"{where}: {cls.__name__} declares no field kinds")
    unknown = sorted(set(section) - set(kinds))
    if unknown:
        raise TrainingSpecError(
            f"{where}: unknown key(s) {', '.join(repr(name) for name in unknown)}"
        )
    values = {}
    for name, raw in section.items():
        converter = _KINDS.get(kinds[name])
        if converter is None:
            raise TrainingSpecError(f"{where}.{name}: unknown declared kind {kinds[name]!r}")
        values[name] = converter(raw, f"{where}.{name}")
    spec = cls(**values)
    issues = spec.issues()
    if issues:
        raise TrainingSpecError(f"{where}: " + "; ".join(issues))
    return spec


@dataclass(frozen=True, slots=True)
class PretrainSpec:
    """RAW Base masked-pretraining hyper-parameters (P1 §2 starting values)."""

    updates: int = 100_000
    effective_batch_frames: int = 256
    micro_batch_frames: int = 16
    lr: float = 1.5e-4
    lr_grid: tuple[float, ...] = (7.5e-5, 1.5e-4, 3e-4)
    weight_decay: float = 0.05
    betas: tuple[float, ...] = (0.9, 0.95)
    warmup_fraction: float = 0.05
    schedule: str = "cosine"
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0
    mask_ratio: float = 0.6
    amp: str = "off"
    platforms: tuple[str, ...] = ("libero", "robotwin")
    checkpoint_every: int = 5_000
    dev_eval_every: int = 5_000
    dev_frames: int = 256
    log_every: int = 50
    dataloader_threads: int = 8

    KINDS: ClassVar[dict[str, str]] = {
        "updates": "int",
        "effective_batch_frames": "int",
        "micro_batch_frames": "int",
        "lr": "float",
        "lr_grid": "floats",
        "weight_decay": "float",
        "betas": "floats",
        "warmup_fraction": "float",
        "schedule": "str",
        "min_lr_ratio": "float",
        "grad_clip": "float",
        "mask_ratio": "float",
        "amp": "str",
        "platforms": "strs",
        "checkpoint_every": "int",
        "dev_eval_every": "int",
        "dev_frames": "int",
        "log_every": "int",
        "dataloader_threads": "int",
    }

    def issues(self) -> list[str]:
        problems: list[str] = []
        if self.updates <= 0:
            problems.append(f"updates must be positive, got {self.updates}")
        if not 0.0 < self.mask_ratio < 1.0:
            problems.append(f"mask_ratio must lie in (0, 1), got {self.mask_ratio}")
        if self.effective_batch_frames <= 0 or self.micro_batch_frames <= 0:
            problems.append("batch sizes must be positive")
        elif self.effective_batch_frames % self.micro_batch_frames:
            problems.append(
                "effective_batch_frames must be a multiple of micro_batch_frames, got "
                f"{self.effective_batch_frames} and {self.micro_batch_frames}"
            )
        if len(self.betas) != 2 or any(value < 0.0 or value >= 1.0 for value in self.betas):
            problems.append(f"betas must be two values in [0, 1), got {tuple(self.betas)!r}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            problems.append(f"warmup_fraction must lie in [0, 1), got {self.warmup_fraction}")
        if self.schedule != "cosine":
            problems.append(f"schedule must be 'cosine', got {self.schedule!r}")
        if self.amp not in ("off", "bf16", "tf32"):
            problems.append(f"amp must be 'off', 'tf32' or 'bf16', got {self.amp!r}")
        if not self.platforms:
            problems.append("platforms must be non-empty")
        if len(self.lr_grid) != 3:
            problems.append(f"lr_grid must declare three scale points, got {tuple(self.lr_grid)!r}")
        return problems

    @property
    def accum_steps(self) -> int:
        return int(self.effective_batch_frames // self.micro_batch_frames)

    def warmup_updates(self, updates: int) -> int:
        return max(1, int(round(self.warmup_fraction * updates))) if updates > 0 else 0


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """Measurement-stage hyper-parameters (P1 §5 starting values)."""

    updates: int = 50_000
    effective_batch_episodes: int = 32
    frames_per_episode: int = 8
    micro_batch_episodes: int = 8
    lr_base: float = 3e-5
    lr_head: float = 3e-4
    lr_grid_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    weight_decay: float = 0.01
    betas: tuple[float, ...] = (0.9, 0.95)
    warmup_fraction: float = 0.05
    schedule: str = "cosine"
    min_lr_ratio: float = 0.0
    grad_clip: float = 1.0
    loss_weights: dict = field(
        default_factory=lambda: {
            "raw_proper_prediction": 1.0,
            "candidate_ce": 1.0,
            "geometry_aux": 0.1,
        }
    )
    sources: tuple[str, ...] = ("synthetic", "real")
    units: str = "raw_dn_uint16_mosaic"
    ce_temperature: float = 1.0
    real_units: str = "normalised_pseudo_raw_float32"
    real_density_verified: bool = False
    synthetic_scenes: int = 32
    synthetic_sensors: tuple = ()
    measurement_model: dict = field(default_factory=lambda: {"kind": "cfa"})
    canvas: tuple = (64, 64)
    action_horizon: int = 4
    grad_conflict_every: int = 5_000
    amp: str = "off"
    checkpoint_every: int = 5_000
    dev_eval_every: int = 5_000
    log_every: int = 50

    KINDS: ClassVar[dict[str, str]] = {
        "updates": "int",
        "effective_batch_episodes": "int",
        "frames_per_episode": "int",
        "micro_batch_episodes": "int",
        "lr_base": "float",
        "lr_head": "float",
        "lr_grid_scales": "floats",
        "weight_decay": "float",
        "betas": "floats",
        "warmup_fraction": "float",
        "schedule": "str",
        "min_lr_ratio": "float",
        "grad_clip": "float",
        "loss_weights": "map",
        "sources": "strs",
        "units": "str",
        "ce_temperature": "float",
        "real_units": "str",
        "real_density_verified": "bool",
        "synthetic_scenes": "int",
        "synthetic_sensors": "raw",
        "measurement_model": "map",
        "canvas": "ints",
        "action_horizon": "int",
        "grad_conflict_every": "int",
        "amp": "str",
        "checkpoint_every": "int",
        "dev_eval_every": "int",
        "log_every": "int",
    }

    def issues(self) -> list[str]:
        problems: list[str] = []
        if self.updates <= 0:
            problems.append(f"updates must be positive, got {self.updates}")
        for name in ("effective_batch_episodes", "frames_per_episode", "micro_batch_episodes"):
            value = getattr(self, name)
            if value <= 0:
                problems.append(f"{name} must be positive, got {value}")
        if self.effective_batch_episodes % self.micro_batch_episodes:
            problems.append(
                "effective_batch_episodes must be a multiple of micro_batch_episodes, got "
                f"{self.effective_batch_episodes} and {self.micro_batch_episodes}"
            )
        if len(self.betas) != 2 or any(value < 0.0 or value >= 1.0 for value in self.betas):
            problems.append(f"betas must be two values in [0, 1), got {tuple(self.betas)!r}")
        if not 0.0 <= self.warmup_fraction < 1.0:
            problems.append(f"warmup_fraction must lie in [0, 1), got {self.warmup_fraction}")
        if self.schedule != "cosine":
            problems.append(f"schedule must be 'cosine', got {self.schedule!r}")
        if self.amp not in ("off", "bf16", "tf32"):
            problems.append(f"amp must be 'off', 'tf32' or 'bf16', got {self.amp!r}")
        if not self.sources:
            problems.append("sources must be non-empty")
        unknown = sorted(set(self.sources) - {"synthetic", "real"})
        if unknown:
            problems.append(f"sources: unknown source(s) {', '.join(unknown)}")
        if self.ce_temperature <= 0.0:
            problems.append(f"ce_temperature must be positive, got {self.ce_temperature}")
        for name in ("raw_proper_prediction", "candidate_ce", "geometry_aux"):
            if name not in self.loss_weights:
                problems.append(f"loss_weights is missing {name!r}")
                continue
            weight = self.loss_weights[name]
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                problems.append(f"loss_weights.{name}: expected a number")
            elif weight < 0.0:
                problems.append(f"loss_weights.{name}: must be non-negative, got {weight}")
        return problems

    @property
    def accum_steps(self) -> int:
        return int(self.effective_batch_episodes // self.micro_batch_episodes)

    def warmup_updates(self, updates: int) -> int:
        return max(1, int(round(self.warmup_fraction * updates))) if updates > 0 else 0


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    every: int = 5_000
    keep_last: int = 3
    save_final: bool = True

    KINDS: ClassVar[dict[str, str]] = {
        "every": "int",
        "keep_last": "int",
        "save_final": "bool",
    }

    def issues(self) -> list[str]:
        problems: list[str] = []
        if self.every <= 0:
            problems.append(f"every must be positive, got {self.every}")
        if self.keep_last < 1:
            problems.append(f"keep_last must be >= 1, got {self.keep_last}")
        return problems


@dataclass(frozen=True, slots=True)
class DataSpec:
    """Frozen P0 assets of one platform: manifest, anchor tables, statistics."""

    cache_root: str = ""
    manifest: str = ""
    manifest_hash: str = ""
    dataset_format: str = ""
    anchors: dict = field(default_factory=dict)
    stats_path: str = ""
    formal_normalisation: str = ""
    window_options: dict = field(default_factory=dict)

    KINDS: ClassVar[dict[str, str]] = {
        "cache_root": "str",
        "manifest": "str",
        "manifest_hash": "str",
        "dataset_format": "str",
        "anchors": "map",
        "stats_path": "str",
        "formal_normalisation": "str",
        "window_options": "map",
    }

    def issues(self) -> list[str]:
        problems: list[str] = []
        for name in ("manifest", "dataset_format"):
            if not getattr(self, name):
                problems.append(f"{name} must be declared")
        for partition in ("train", "dev"):
            if partition not in self.anchors:
                problems.append(f"anchors is missing partition {partition!r}")
        return problems

    def anchor_path(self, partition: str) -> str:
        entry = self.anchors.get(partition)
        if isinstance(entry, Mapping):
            return str(entry.get("path", ""))
        return str(entry or "")

    def anchor_sha256(self, partition: str) -> str:
        entry = self.anchors.get(partition)
        if isinstance(entry, Mapping):
            return str(entry.get("json_sha256") or entry.get("gzip_sha256") or "")
        return ""


# ---------------------------------------------------------------------------
# run context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunContext:
    """Where one run writes and what identifies it."""

    stage: str
    seed: int
    run_id: str
    run_dir: Path
    config_path: str
    config_hash: str
    command: tuple[str, ...]
    tag: str = ""

    @property
    def checkpoints(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def train_log(self) -> Path:
        return self.run_dir / "train_log.jsonl"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": RUN_MANIFEST_FORMAT,
            "stage": self.stage,
            "seed": int(self.seed),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "command": list(self.command),
            "tag": self.tag,
        }


def build_run_context(
    args: argparse.Namespace, cfg: n02_config.N02FullConfig, *, command: Sequence[str]
) -> RunContext:
    stage = str(args.stage)
    seed = int(args.seed)
    tag = str(getattr(args, "tag", "") or "")
    suffix = f"-{tag}" if tag else ""
    run_id = f"{stage}-seed{seed}{suffix}"
    root = (
        Path(args.output)
        if args.output
        else Path(str(cfg.paths.get("output_root", DEFAULT_OUTPUT_ROOT)))
    )
    return RunContext(
        stage=stage,
        seed=seed,
        run_id=run_id,
        run_dir=root / run_id,
        config_path=str(args.config),
        config_hash=n02_config.config_hash(cfg),
        command=tuple(str(item) for item in command),
        tag=tag,
    )


# ---------------------------------------------------------------------------
# checkpoints and trainer state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainerState:
    """Resumable position and bookkeeping of one training run."""

    updates: int = 0
    micro: int = 0
    updates_this_resume: int = 0
    resume_updates: int = 0
    history: list[dict] = field(default_factory=list)
    checkpoint_files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "updates": int(self.updates),
            "micro": int(self.micro),
            "updates_this_resume": int(self.updates_this_resume),
            "resume_updates": int(self.resume_updates),
        }


def capture_rng_state() -> dict[str, Any]:
    torch = _torch()
    np = _numpy()
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            payload["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:  # noqa: BLE001 - a CUDA host without a context is not a failure
            payload["cuda"] = None
    return payload


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    torch = _torch()
    np = _numpy()
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if payload.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate one checkpoint container."""
    torch = _torch()
    target = Path(path)
    if not target.is_file():
        raise TrainNotAvailable(f"checkpoint not found: {target}", missing=(str(target),))
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise TrainNotAvailable(
            f"{target} is not a {CHECKPOINT_FORMAT} checkpoint", missing=(str(target),)
        )
    return payload


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def lr_multiplier(update: int, *, updates: int, warmup: int, min_ratio: float = 0.0) -> float:
    """Linear warmup then cosine decay; a multiplier of the peak learning rate."""
    if updates <= 0:
        return 1.0
    if update < warmup:
        return float(update + 1) / float(max(1, warmup))
    span = max(1, updates - warmup)
    progress = min(1.0, max(0.0, (update - warmup) / float(span)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


def apply_lr(
    optimizer: Any,
    base_lrs: Sequence[float],
    *,
    update: int,
    updates: int,
    warmup: int,
    min_ratio: float = 0.0,
) -> list[float]:
    multiplier = lr_multiplier(update, updates=updates, warmup=warmup, min_ratio=min_ratio)
    current: list[float] = []
    for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = float(base) * multiplier
        current.append(float(group["lr"]))
    return current


# ---------------------------------------------------------------------------
# parameter roles and gradient statistics
# ---------------------------------------------------------------------------

_HEAD_HINTS = ("head", "decoder", "predictor", "projection", "projector", "candidate")


def split_parameter_names(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split parameter names into (base, heads) by an explicit role marker.

    A name without a marker counts as Base, which is the conservative reading: an
    unmarked head is reported under Base rather than inflating the head column.
    """
    base: list[str] = []
    heads: list[str] = []
    for name in names:
        lowered = name.lower()
        (heads if any(hint in lowered for hint in _HEAD_HINTS) else base).append(name)
    return base, heads


def model_named_parameters(model: Any) -> list[tuple[str, Any]]:
    """Named parameters of an ``nn.Module`` or of the T3 ``ObservationModel`` façade.

    ``ObservationModel`` is not an ``nn.Module``; it holds its trainable parts in
    ``encoder``/``decoder``/``candidate_encoder`` attributes, so the harness reads
    them with the module name as the parameter-name prefix (which keeps the
    Base/heads split by the declared role markers meaningful).
    """
    direct = getattr(model, "named_parameters", None)
    if callable(direct):
        return list(direct())
    pairs: list[tuple[str, Any]] = []
    for attribute in ("encoder", "decoder", "candidate_encoder"):
        module = getattr(model, attribute, None)
        if callable(getattr(module, "named_parameters", None)):
            pairs.extend(
                (f"{attribute}.{name}", value) for name, value in module.named_parameters()
            )
    return pairs


def model_parameters(model: Any) -> list[Any]:
    return [value for _, value in model_named_parameters(model)]


def _nested_state(model: Any, flat: Mapping[str, Any]) -> Any:
    """Rebuild the nested state dict a facade's ``load_state_dict`` expects."""
    if callable(getattr(model, "load_state_dict", None)) and hasattr(model, "state_dict"):
        return dict(flat)
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        head, _, tail = key.partition(".")
        if not tail:
            nested[key] = value
            continue
        nested.setdefault(head, {})[tail] = value
    return nested


def flatten_state_dict(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a (possibly nested) state dict to ``name -> tensor`` pairs."""
    flat: dict[str, Any] = {}
    for key, value in dict(payload).items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_state_dict(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def model_state_dict(model: Any) -> Any:
    """State dict of an ``nn.Module`` or of the T3 facade's trainable submodules."""
    direct = getattr(model, "state_dict", None)
    if callable(direct):
        return direct()
    return {
        attribute: getattr(model, attribute).state_dict()
        for attribute in ("encoder", "decoder", "candidate_encoder")
        if callable(getattr(getattr(model, attribute, None), "state_dict", None))
    }


def _base_key_candidates(key: str) -> list[str]:
    """Key spellings that connect a pretrainer's Base to the measurement model's.

    The FCMAE wraps the raw backbone directly (``fcmae.encoder.downsample_layers.*``)
    while the measurement model holds a ``RawObservationEncoder`` around it
    (``encoder.backbone.downsample_layers.*``), so the mapping is tried in both
    directions rather than assumed.
    """
    candidates = [key]
    if key.startswith("fcmae.encoder."):
        rest = key[len("fcmae.encoder.") :]
        candidates += [
            "encoder." + rest,
            "encoder.backbone." + rest,
            "encoder.encoder." + rest,
        ]
    if key.startswith("encoder.backbone."):
        rest = key[len("encoder.backbone.") :]
        candidates += ["fcmae.encoder." + rest, "encoder." + rest]
    elif key.startswith("encoder."):
        rest = key[len("encoder.") :]
        candidates += ["fcmae.encoder." + rest]
    return candidates


def load_base_subset(model: Any, payload: Any) -> dict[str, Any]:
    """Load the Base weights a pretrainer checkpoint contributes, with a full report.

    A pretraining checkpoint holds ``fcmae.encoder.*`` plus pretraining-only modules
    (``mask_token``, the FCMAE decoder); the measurement model holds ``encoder.*`` and
    its own heads. This maps the shared Base tensor-for-tensor, reports every skipped
    pretraining module and the matched/unmatched counts, and refuses to report success
    when no Base tensor was loaded -- the caller can then tell a real inheritance from
    a silent no-op.
    """
    torch = _torch()
    model_state = flatten_state_dict(model_state_dict(model))
    payload_state = flatten_state_dict(payload)
    mapped: dict[str, Any] = {}
    skipped: list[str] = []
    shape_conflicts: list[str] = []
    for key, value in payload_state.items():
        target = next((name for name in _base_key_candidates(key) if name in model_state), None)
        if target is None:
            skipped.append(key)
            continue
        existing = model_state[target]
        if (
            hasattr(existing, "shape")
            and hasattr(value, "shape")
            and tuple(existing.shape) != tuple(value.shape)
        ):
            shape_conflicts.append(
                f"{key} -> {target} {tuple(value.shape)} vs {tuple(existing.shape)}"
            )
            continue
        if not torch.is_tensor(existing) or not torch.is_tensor(value):
            continue
        mapped[target] = value
    if not mapped:
        raise TrainNotAvailable(
            "the checkpoint contributes no Base tensor to this model; refusing to report a "
            f"successful inheritance (skipped {len(skipped)} pretraining tensors, "
            f"{len(shape_conflicts)} shape conflicts)"
        )
    current = model_state_dict(model)
    load_model_state_dict(model, _merge_state(current, mapped), strict=False)
    unmatched_model = sorted(name for name in model_state if name not in mapped)
    return {
        "loaded_tensors": len(mapped),
        "pretraining_tensors": len(payload_state),
        "skipped_pretraining_tensors": len(skipped),
        "skipped_modules": sorted({name.split(".")[0] for name in skipped}),
        "shape_conflicts": shape_conflicts[:8],
        "unmatched_model_tensors": len(unmatched_model),
        "unmatched_model_prefixes": sorted(
            {
                name.split(".")[0] + "." + (name.split(".")[1] if "." in name else "")
                for name in unmatched_model
            }
        )[:8],
        "base_loaded": True,
    }


def _merge_state(current: Any, updates: Mapping[str, Any]) -> Any:
    """Overlay flat updates onto a (possibly nested) state dict, keeping the layout."""
    if isinstance(current, Mapping) and not hasattr(current, "state_dict"):
        nested: dict[str, Any] = {}
        for key, value in dict(current).items():
            if isinstance(value, Mapping):
                prefix = f"{key}."
                inner = {
                    name[len(prefix) :]: item
                    for name, item in updates.items()
                    if name.startswith(prefix)
                }
                nested[key] = _merge_state(value, inner) if inner else value
            elif key in updates:
                nested[key] = updates[key]
            else:
                nested[key] = value
        return nested
    return dict(updates)


def load_model_state_dict(model: Any, payload: Any, *, strict: bool = True) -> None:
    direct = getattr(model, "load_state_dict", None)
    if callable(direct):
        direct(payload, strict=strict)
        return
    if not isinstance(payload, Mapping):
        raise TrainNotAvailable("facade checkpoint does not hold a per-module state dict")
    for attribute, state in payload.items():
        module = getattr(model, attribute, None)
        if not callable(getattr(module, "load_state_dict", None)):
            raise TrainNotAvailable(f"checkpoint declares unknown submodule {attribute!r}")
        module.load_state_dict(state, strict=strict)


def move_model_to(model: Any, device: str) -> Any:
    """Move an ``nn.Module`` or the T3 façade's submodules to ``device``."""
    if hasattr(model, "to") and hasattr(model, "named_parameters"):
        return model.to(device)
    for attribute in ("encoder", "decoder", "candidate_encoder"):
        module = getattr(model, attribute, None)
        if callable(getattr(module, "to", None)):
            module.to(device)
    return model


def _grad_vector(parameters: Iterable[Any]) -> Any:
    torch = _torch()
    parts = [p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    if not parts:
        return None
    return torch.cat(parts)


def gradient_statistics(model: Any) -> dict[str, Any]:
    """Base/head gradient norms, per group.

    The angle between the Base and head gradient vectors is *not* defined: they
    are gradients with respect to parameters of different shape, so the dot
    product the angle needs does not exist. This function therefore reports the
    two norms (and their ratio) and leaves ``grad_angle_deg`` null; the defined
    angle is the per-term gradient conflict computed by
    :func:`term_gradient_conflict` over the shared parameter vector.
    """
    named = model_named_parameters(model)
    base_names, head_names = split_parameter_names([name for name, _ in named])
    by_name = dict(named)
    base = _grad_vector([by_name[name] for name in base_names])
    heads = _grad_vector([by_name[name] for name in head_names])
    base_norm = None if base is None else float(base.norm().item())
    head_norm = None if heads is None else float(heads.norm().item())
    payload: dict[str, Any] = {
        "grad_norm_base": base_norm,
        "grad_norm_heads": head_norm,
        "grad_norm_ratio_heads_over_base": (
            None if not base_norm or head_norm is None else head_norm / base_norm
        ),
        "grad_angle_deg": None,
        "grad_angle_definition": "undefined between different parameter spaces; see term angles",
        "base_parameter_count": len(base_names),
        "head_parameter_count": len(head_names),
    }
    return payload


def term_gradient_conflict(model: Any, term_tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Pairwise cosine between the gradients of the *differentiable* loss terms.

    All terms are differentiated with respect to the same full parameter vector,
    so each cosine (and its angle) is well defined on shared parameters. This is
    the quantity the "Base/head gradient conflict" diagnostic needs; it is
    computed only at the declared interval because each term costs one extra
    backward pass.
    """
    torch = _torch()
    parameters = [p for p in model_parameters(model) if p.requires_grad]
    if not parameters or len(term_tensors) < 2:
        return {"term_angles": {}, "term_gradient_norms": {}}
    vectors: dict[str, Any] = {}
    norms: dict[str, float] = {}
    for name, tensor in term_tensors.items():
        if tensor is None or not torch.is_tensor(tensor) or not tensor.requires_grad:
            continue
        try:
            grads = torch.autograd.grad(tensor, parameters, retain_graph=True, allow_unused=True)
        except RuntimeError as exc:  # pragma: no cover - a disconnected term is recorded
            norms[name] = float("nan")
            del exc
            continue
        parts = [
            (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
            for grad, param in zip(grads, parameters, strict=True)
        ]
        vectors[name] = torch.cat(parts)
        norms[name] = float(vectors[name].norm().item())
    angles: dict[str, float | None] = {}
    cosines: dict[str, float | None] = {}
    names = sorted(vectors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            denominator = norms[left] * norms[right]
            cosine = None
            if denominator > 0.0:
                cosine = float(torch.dot(vectors[left], vectors[right]).item()) / denominator
                cosine = max(-1.0, min(1.0, cosine))
            cosines[f"{left}|{right}"] = cosine
            angles[f"{left}|{right}"] = None if cosine is None else math.degrees(math.acos(cosine))
    return {"term_angles": angles, "term_cosines": cosines, "term_gradient_norms": norms}


# ---------------------------------------------------------------------------
# frame batching
# ---------------------------------------------------------------------------


#: Frame fields that carry the frozen window's unit outer (time) axis.
_UNIT_AXIS_FIELDS = ("values", "valid_mask", "bad_mask", "saturated_mask")


def stack_measurement_frames(
    frames: Sequence[Any], *, camera_id: str | None = None, squeeze_leading: bool = False
) -> Any:
    """Stack one-camera single-sample frames into one batched frame.

    Values and masks gain a leading batch axis; per-row identity stays in
    ``episode_ids``/``frame_ids``. ``metadata`` keeps the shared schema of the
    first row (rows of one generated family) and records the batch width;
    ``metadata_available`` is the conjunction over rows, so a field absent in any
    row is reported absent rather than guessed.

    ``squeeze_leading`` drops a unit outer axis (the frozen anchor window is
    ``rnn_frames=1``, so the loader's frames arrive as ``[1, C, H, W]``) before
    stacking, giving the ``[B, C, H, W]`` a batched encoder expects.
    """
    np = _numpy()
    if not frames:
        raise ValueError("stack_measurement_frames needs at least one frame")
    from owac.core.encoder_contract import MeasurementFrame

    def _array(frame: Any, name: str) -> Any:
        array = np.asarray(getattr(frame, name))
        if (
            squeeze_leading
            and name in _UNIT_AXIS_FIELDS
            and array.ndim >= 3
            and array.shape[0] == 1
        ):
            array = array[0]
        return array

    def _stack(name: str) -> Any:
        arrays = [getattr(frame, name) for frame in frames]
        if all(array is None for array in arrays):
            return None
        if any(array is None for array in arrays):
            raise ValueError(f"frame field {name!r} is present in only some rows")
        return np.stack([_array(frame, name) for frame in frames], axis=0)

    metadata = dict(frames[0].metadata)
    metadata["batch_rows"] = len(frames)
    values = _stack("values")
    if values is not None and np.issubdtype(values.dtype, np.integer):
        # The online path indexes the raw grid before casting (model._build_raw_query),
        # and PyTorch has no integer advanced-indexing kernel. Casting here keeps every
        # DN value exact in float32 and leaves the declared unit string untouched; the
        # original storage dtype is recorded so the receipt still says what was stored.
        metadata["value_dtype_original"] = str(values.dtype)
        values = values.astype(np.float32)
    available: dict[str, Any] = {}
    for frame in frames:
        for key, value in frame.metadata_available.items():
            available[key] = bool(available.get(key, True)) and bool(value)
    episode_ids = tuple(str(item) for frame in frames for item in frame.episode_ids)
    frame_ids = tuple(str(item) for frame in frames for item in frame.frame_ids)
    return MeasurementFrame(
        values=values,
        valid_mask=_stack("valid_mask"),
        bad_mask=_stack("bad_mask"),
        saturated_mask=_stack("saturated_mask"),
        sample_xy=_stack("sample_xy"),
        episode_ids=episode_ids,
        frame_ids=frame_ids,
        camera_id=str(camera_id or frames[0].camera_id),
        timestamp=None,
        time_unit=frames[0].time_unit,
        new_measurement=np.stack(
            [np.asarray(frame.is_new_measurement).reshape(()) for frame in frames]
        ),
        metadata=metadata,
        metadata_available=available,
    )


def merge_bundles(bundles: Sequence[Any]) -> Any:
    """Merge per-episode bundles into one bundle whose batch axis is the episodes."""
    np = _numpy()
    from owac.core.encoder_contract import EmbodimentContext

    if not bundles:
        raise ValueError("no bundles to merge")
    if len(bundles) == 1:
        # The online path keeps single-sample frames unbatched (model._is_batched_frame
        # returns False when the bundle batch is 1), so a one-episode batch must not
        # grow a leading batch axis.
        bundle = bundles[0]
        return dataclasses.replace(
            bundle,
            frames={camera: _float32_frame(frame) for camera, frame in bundle.frames.items()},
            new_time_mask=(
                bundle.new_time_mask if bundle.new_time_mask is not None else np.ones(1, dtype=bool)
            ),
        )
    camera_ids = tuple(bundles[0].frames)
    # Every row of one batched decision must be a distinct acquisition: the ledger
    # keys evidence by acquisition_id, so merging duplicate frame identities would
    # silently collapse independent scenes into one measurement.
    seen: dict[tuple[str, str], int] = {}
    duplicates: list[str] = []
    for bundle in bundles:
        for camera_id in camera_ids:
            frame = bundle.frames[camera_id]
            for row, source_frame_id in enumerate(frame.frame_ids):
                episode = frame.episode_ids[row] if row < len(frame.episode_ids) else ""
                key = (str(camera_id), str(episode), str(source_frame_id))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] == 2:
                    duplicates.append(":".join(key))
    if duplicates:
        unique = sorted(set(duplicates))
        raise TrainNotAvailable(
            f"merged batch would repeat {len(unique)} acquisition identities "
            f"(for example {', '.join(unique[:3])}); the measurement ledger would "
            "deduplicate distinct scenes into one piece of evidence"
        )
    frames = {
        camera_id: stack_measurement_frames(
            [bundle.frames[camera_id] for bundle in bundles], camera_id=camera_id
        )
        for camera_id in camera_ids
    }
    first_context = bundles[0].context
    context = EmbodimentContext(
        proprio=_stack_optional([bundle.context.proprio for bundle in bundles]),
        proprio_available=bool(all(bool(bundle.context.proprio_available) for bundle in bundles)),
        executed_action=_stack_optional([bundle.context.executed_action for bundle in bundles]),
        executed_action_available=bool(
            all(bool(bundle.context.executed_action_available) for bundle in bundles)
        ),
        camera_transform=None,
        camera_transform_available=False,
        task_tokens=first_context.task_tokens,
        task_mask=first_context.task_mask,
        hypothesis_prior=first_context.hypothesis_prior,
        hypothesis_prior_available=first_context.hypothesis_prior_available,
    )
    return dataclasses.replace(
        bundles[0],
        frames=frames,
        context=context,
        episode_ids=tuple(str(item) for bundle in bundles for item in bundle.episode_ids),
        anchor_ids=tuple(str(item) for bundle in bundles for item in bundle.anchor_ids),
        new_time_mask=np.ones(len(bundles), dtype=bool),
    )


def _float32_frame(frame: Any) -> Any:
    """Integer raw values as float32 (exact in this range) with the dtype recorded.

    ``model._build_raw_query`` indexes the raw grid before casting and PyTorch has
    no integer advanced-indexing kernel; the DN values and the declared unit string
    are unchanged, and the original storage dtype stays in the frame metadata.
    """
    np = _numpy()
    values = np.asarray(frame.values)
    if not np.issubdtype(values.dtype, np.integer):
        return frame
    metadata = dict(frame.metadata)
    metadata["value_dtype_original"] = str(values.dtype)
    return dataclasses.replace(frame, values=values.astype(np.float32), metadata=metadata)


def _stack_optional(values: Sequence[Any]) -> Any:
    np = _numpy()
    if not values or any(value is None for value in values):
        return None
    try:
        return np.stack([np.asarray(value) for value in values], axis=0)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# loss composition (harness side, interface freeze §3.7)
# ---------------------------------------------------------------------------


def _term(
    status: str,
    *,
    value: float | None = None,
    total: float | None = None,
    activations: int = 0,
    reason: str = "",
    kind: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "value": value,
        "sum": total,
        "activation_count": int(activations),
        "reason": reason,
    }
    if kind:
        payload["kind"] = kind
    return payload


def _candidate_index(targets: Any) -> tuple[int | None, str]:
    """The explicit correct-candidate label written by the task constructor."""
    geometry = getattr(targets, "optional_geometry", None)
    if isinstance(geometry, Mapping) and "candidate_index" in geometry:
        value = geometry["candidate_index"]
        if isinstance(value, bool) or not isinstance(value, int):
            return None, f"candidate_index is {type(value).__name__}, expected int"
        return int(value), "TrainingTargets.optional_geometry['candidate_index']"
    return None, (
        "TrainingTargets.optional_geometry carries no candidate_index "
        "(an all-wrong dictionary or a real window has no correct candidate)"
    )


def _check_kind(score: Any) -> str:
    """``proper`` (normalised density) or ``empirical`` (a check that is not proper)."""
    stats = getattr(score, "local_stats", None)
    if isinstance(stats, Mapping):
        marker = stats.get("calibration_only")
        empirical = stats.get("empirical_check")
        if marker is not None or empirical is not None:
            flagged = bool(np_bool(marker)) or bool(np_bool(empirical))
            if flagged:
                return "empirical"
    kind = str(getattr(score, "kind", ""))
    if kind == "empirical_energy":
        return "empirical"
    return "proper"


def np_bool(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except Exception:  # noqa: BLE001 - a non-scalar marker counts as set
            return True
    return bool(value)


def _scale_suspect(score: Any, *, threshold: float = 1e3) -> bool | None:
    """``True`` when the median absolute standardised residual exceeds ``threshold``."""
    stats = getattr(score, "local_stats", None)
    if not isinstance(stats, Mapping) or "standardised_residual" not in stats:
        return None
    value = stats["standardised_residual"]
    if hasattr(value, "detach"):
        value = value.detach()
    try:
        residual = _numpy().asarray(value, dtype="float64").reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None
    if residual.size == 0:
        return None
    return bool(float(_numpy().median(_numpy().abs(residual))) > float(threshold))


def _check_columns(score: Any) -> tuple[Any, Any, int, str]:
    """``(per_valid_scalar, raw_nll_or_energy, activation_count, kind)``."""
    torch = _torch()
    per = getattr(score, "per_valid_scalar", None)
    raw = getattr(score, "raw_nll_or_energy", None)
    if per is None:
        per = raw
    if per is None:
        raise TrainNotAvailable(
            "MeasurementScore carries neither per_valid_scalar nor raw_nll_or_energy"
        )
    per_tensor = per if torch.is_tensor(per) else torch.tensor(float(per))
    if torch.is_tensor(raw):
        raw_tensor = raw
    elif raw is None:
        raw_tensor = per_tensor
    else:
        raw_tensor = torch.tensor(float(raw))
    if not torch.isfinite(per_tensor.reshape(())) or not torch.isfinite(raw_tensor.reshape(())):
        raise TrainNotAvailable("proper score is not finite")
    count = 0
    denominator = float(per_tensor.detach().reshape(()).item())
    if denominator not in (0.0,):
        count = max(0, int(round(float(raw_tensor.detach().reshape(()).item()) / denominator)))
    return per_tensor.reshape(()), raw_tensor.reshape(()), count, _check_kind(score)


def _world_selection(
    scores: Sequence[Any], worlds: Sequence[Any], step: Any, label: int | None
) -> tuple[Any | None, str, str]:
    """Pick the world whose proper score trains the decoder.

    ``candidate_index`` is the supervised choice when the task provides one; otherwise
    the step's own ``best_world`` is used and the choice is recorded, so an unlabelled
    real window still trains a declared explanation instead of silently averaging the
    dictionary. With no candidate at all the term is ``unavailable``.
    """
    if not scores:
        return None, "unavailable", "no score was produced"
    if label is not None and 0 <= label < len(scores):
        return scores[label], "candidate_index", ""
    best = getattr(step, "best_world", None)
    if best is not None:
        for index, world in enumerate(worlds):
            if world is best and index < len(scores):
                return scores[index], "best_world", ""
    if len(scores) == 1:
        return scores[0], "single_world", ""
    return (
        None,
        "unavailable",
        (f"no candidate_index and no identifiable best_world among {len(scores)} worlds"),
    )


def _proper_term(
    name: str,
    scores: Sequence[Any],
    worlds: Sequence[Any],
    step: Any,
    label: int | None,
    weights: Mapping[str, float],
) -> tuple[Any | None, dict[str, Any]]:
    selected, selection, reason = _world_selection(scores, worlds, step, label)
    if selected is None:
        return None, _term("unavailable", reason=reason or f"{name} has no usable score")
    per, raw, count, kind = _check_columns(selected)
    record = _term(
        "ok",
        value=float(per.detach().item()),
        total=float(raw.detach().item()),
        activations=count,
        kind=f"proper_nll_{name}" if kind == "proper" else f"empirical_energy_{name}",
    )
    record["score_selection"] = selection
    record["world_count"] = len(scores)
    record["check_kind"] = kind
    # T2's cheap scale detector, computed from the column the checks already carry:
    # a median |standardised residual| above 1e3 is a unit/scale mistake, not a wrong
    # candidate.
    record["scale_suspect"] = _scale_suspect(selected)
    record["calibration_only"] = bool(
        np_bool(
            (getattr(selected, "local_stats", None) or {}).get("calibration_only")
            if isinstance(getattr(selected, "local_stats", None), Mapping)
            else False
        )
    )
    record["tensor"] = per
    if float(weights.get(name, 1.0)) == 0.0:
        return None, record
    return per, record


def geometry_aux_term(best_world: Any, targets: Any) -> tuple[Any | None, dict[str, Any]]:
    """MSE between the best world's view-2d geometry and the truth geometry."""
    torch = _torch()
    geometry = getattr(targets, "optional_geometry", None)
    if not isinstance(geometry, Mapping) or "world_px" not in geometry:
        return None, _term(
            "unavailable", reason="optional_geometry has no 'world_px' truth geometry"
        )
    if best_world is None:
        return None, _term("unavailable", reason="step has no best_world")
    frame = str(getattr(best_world, "geometry_frame", "view_2d"))
    if frame not in ("", "view_2d"):
        return None, _term(
            "unavailable", reason=f"best_world.geometry_frame={frame!r} is not view_2d"
        )
    truth = geometry["world_px"]
    truth_tensor = truth if torch.is_tensor(truth) else torch.as_tensor(_numpy().asarray(truth))
    truth_tensor = truth_tensor.to(dtype=torch.float32).reshape(-1, 2)
    np = _numpy()
    rows = []
    skipped = 0
    for entity in tuple(getattr(best_world, "entities", ()) or ()):
        value = getattr(entity, "geometry", None)
        if value is None:
            skipped += 1
            continue
        if torch.is_tensor(value):
            rows.append(value.reshape(-1).to(dtype=torch.float32)[:2])
            continue
        try:
            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            # An entity's geometry may be a declared structure the harness cannot
            # treat as pixel coordinates; it is skipped and reported, never forced.
            skipped += 1
            continue
        if array.size < 2:
            skipped += 1
            continue
        rows.append(torch.as_tensor(array[:2], dtype=torch.float32))
    if not rows:
        return None, _term(
            "unavailable",
            reason=f"best_world has no usable entity geometry ({skipped} row(s) skipped)",
        )
    try:
        stacked = torch.stack([row.reshape(2) for row in rows])
    except (RuntimeError, ValueError) as exc:
        return None, _term("unavailable", reason=f"entity geometry is not 2D: {exc}")
    count = int(min(stacked.shape[0], truth_tensor.shape[0]))
    if count == 0:
        return None, _term("unavailable", reason="no overlapping entity rows")
    difference = stacked[:count] - truth_tensor[:count]
    value = (difference * difference).mean()
    scalar = float(value.detach().item())
    record = _term(
        "ok",
        value=scalar,
        total=float((difference * difference).sum().detach().item()),
        activations=count * 2,
        kind="mse_view_2d",
    )
    if skipped:
        record["skipped_rows"] = skipped
    return value, record


def loss_terms_from_step(
    step: Any,
    targets: Any | None,
    *,
    weights: Mapping[str, float] | None = None,
    ce_temperature: float = 1.0,
) -> tuple[Any | None, dict[str, dict[str, Any]]]:
    """Compose the declared P1 loss terms from one ``StepOutput``.

    Returns ``(total, terms)``. A term without its explicit label or truth
    geometry is ``unavailable`` with a reason and contributes nothing; it is never
    written as a zero-valued success. The prequential and fit proper scores are
    separate columns of the one proper-prediction target and are never summed into
    a second likelihood.
    """
    torch = _torch()
    # The fit column is monitoring only: the one proper-prediction target keeps
    # counting once, so its default weight is zero unless a caller overrides it.
    weights = {"raw_proper_prediction_fit": 0.0, **dict(weights or {})}
    terms: dict[str, dict[str, Any]] = {}
    total: Any = None

    def _add(name: str, value: Any | None, record: dict[str, Any]) -> None:
        nonlocal total
        if value is not None:
            record["tensor"] = value
        terms[name] = record
        if value is None:
            return
        weight = float(weights.get(name, 1.0))
        if weight == 0.0:
            return
        total = value * weight if total is None else total + value * weight

    worlds = tuple(getattr(step, "worlds", ()) or ())
    label, label_source = _candidate_index(targets)
    locked = tuple(getattr(step, "locked_checks", ()) or ())
    fit_scores = tuple(getattr(step, "fit_scores", ()) or ())

    value, record = _proper_term(
        "raw_proper_prediction_prequential", locked, worlds, step, label, weights
    )
    _add("raw_proper_prediction_prequential", value, record)

    value, record = _proper_term(
        "raw_proper_prediction_fit", fit_scores, worlds, step, label, weights
    )
    _add("raw_proper_prediction_fit", value, record)

    if label is None:
        _add("candidate_ce", None, _term("unavailable", reason=label_source))
    elif not worlds:
        _add("candidate_ce", None, _term("unavailable", reason="step carries no world candidates"))
    elif not 0 <= label < len(worlds):
        _add(
            "candidate_ce",
            None,
            _term("unavailable", reason=f"candidate_index {label} is outside {len(worlds)} worlds"),
        )
    else:
        temperature = float(ce_temperature)
        world_scores = getattr(step, "world_scores", None)
        if torch.is_tensor(world_scores) and world_scores.numel() == len(worlds):
            # T3 exposes the differentiable score tensor; softmax(-score / T) treats the
            # score as a cost (lower is better), which is the declared convention.
            storage = "differentiable"
            scores = -world_scores.reshape(-1).to(torch.float32) / temperature
        else:
            values = [world.score for world in worlds]
            storage = "tensor" if all(torch.is_tensor(value) for value in values) else "constant"
            scores = (
                -torch.stack([value.reshape(()) for value in values]) / temperature
                if storage == "tensor"
                else -torch.tensor([float(value) for value in values], dtype=torch.float32)
                / temperature
            )
        kinds = {str(getattr(world, "score_kind", "")) for world in worlds}
        if kinds == {"normalized_density"}:
            cross_entropy, kind = (
                -torch.log_softmax(scores, dim=0)[label],
                ("cross_entropy_normalized_density"),
            )
        elif kinds == {"empirical_energy"}:
            cross_entropy, kind = (
                -torch.log_softmax(-scores, dim=0)[label],
                ("cross_entropy_empirical_energy"),
            )
        else:
            cross_entropy, kind = None, ""
        if cross_entropy is None:
            _add(
                "candidate_ce",
                None,
                _term(
                    "unavailable",
                    reason=f"world score kinds are not one consistent family: {sorted(kinds)}",
                ),
            )
        else:
            record = _term(
                "ok",
                value=float(cross_entropy.detach().item()),
                total=float(cross_entropy.detach().item()),
                activations=1,
                kind=kind,
            )
            record["label_source"] = label_source
            record["score_storage"] = storage
            record["temperature"] = temperature
            record["score_convention"] = "softmax(-world_score / temperature)"
            record["score_kind"] = sorted(kinds)[0]
            record["gradient_available"] = storage in ("tensor", "differentiable")
            if storage == "constant":
                record["status"] = "detached"
                cross_entropy = cross_entropy.detach()
            _add("candidate_ce", cross_entropy, record)

    value, record = geometry_aux_term(getattr(step, "best_world", None), targets)
    _add("geometry_aux", value, record)
    return total, terms


# ---------------------------------------------------------------------------
# trainer core
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MicroInfo:
    update: int
    micro: int
    seed: int


class P1Trainer:
    """Gradient-accumulated training loop with exact resume and instrumentation.

    The trainer is model-agnostic: ``loss_fn(model, batch, info)`` returns
    ``(loss, term_records, batch_info)``, so the same loop drives FCMAE
    pretraining and the measurement stage, and the tests drive it with a tiny
    deterministic model.
    """

    def __init__(
        self,
        *,
        stage: str,
        seed: int,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        source: Any,
        loss_fn: Any,
        updates: int,
        accum_steps: int,
        base_lrs: Sequence[float],
        warmup_updates: int,
        min_lr_ratio: float = 0.0,
        clip_grad: float = 1.0,
        device: str = "cpu",
        amp: str = "off",
        history_limit: int = 20_000,
        grad_conflict_every: int = 0,
    ) -> None:
        self.stage = str(stage)
        self.seed = int(seed)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.source = source
        self.loss_fn = loss_fn
        self.updates = int(updates)
        self.accum_steps = int(accum_steps)
        self.base_lrs = [float(value) for value in base_lrs]
        self.warmup_updates = int(warmup_updates)
        self.min_lr_ratio = float(min_lr_ratio)
        self.clip_grad = float(clip_grad)
        self.device = str(device)
        self.amp = str(amp)
        self.resumed_from_stage = ""
        self.base_inheritance_report: dict[str, Any] = {}
        self.history_limit = int(history_limit)
        self.grad_conflict_every = int(grad_conflict_every)
        self.state = TrainerState()
        self._schedule_lr()

    # -- position ---------------------------------------------------------------

    def sampler_state(self) -> dict[str, Any]:
        return {"position": self.source.state_dict()}

    def load_sampler_state(self, payload: Mapping[str, Any]) -> None:
        self.source.load_state_dict(dict(payload.get("position", payload)))

    def checkpoint_payload(
        self,
        *,
        partition: Mapping[str, Any] | None = None,
        config_hash: str = "",
        run_id: str = "",
        extras: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "stage": self.stage,
            "seed": self.seed,
            "run_id": run_id,
            "saved_at": _now(),
            "updates": int(self.state.updates),
            "model": model_state_dict(self.model),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "amp": {
                "scaler": None if self.scaler is None else self.scaler.state_dict(),
                "enabled": self.scaler is not None,
            },
            "rng": capture_rng_state(),
            "sampler": self.sampler_state(),
            "partition": dict(partition or {}),
            "config_hash": config_hash,
            "versions": interpreter_info(),
            "extras": dict(extras or {}),
        }

    def save(self, path: str | Path, **kwargs: Any) -> Path:
        torch = _torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(**kwargs), target)
        self.state.checkpoint_files.append(str(target))
        return target

    def resume_from(self, payload: Mapping[str, Any], *, strict: bool = True) -> None:
        """Restore every resume-relevant quantity from one checkpoint payload."""
        source_stage = payload.get("stage")
        base_inheritance = source_stage == "p1-pretrain" and self.stage == "p1-measurement"
        if source_stage not in (None, self.stage) and not base_inheritance:
            raise TrainNotAvailable(
                f"checkpoint stage {source_stage!r} cannot be resumed as {self.stage!r}"
            )
        self.resumed_from_stage = str(source_stage or "")
        self.base_inheritance_report = {}
        if base_inheritance:
            # The P1 main flow inherits the Base from the pretraining checkpoint and
            # starts the measurement heads fresh; the report records exactly what was
            # inherited and what the pretraining-only modules were. The pretraining
            # optimizer/scheduler/AMP/RNG/data position belong to that run and are
            # deliberately not resumed, so this run's update counters start at zero
            # while the source update count stays on the receipt.
            self.base_inheritance_report = load_base_subset(self.model, payload["model"])
            self.base_inheritance_report["source_updates"] = int(payload.get("updates", 0))
            self.base_inheritance_report["source_checkpoint"] = str(payload.get("run_id", ""))
            self.state.updates = 0
            self.state.micro = 0
            self.state.resume_updates = 0
            self.state.updates_this_resume = 0
            self._schedule_lr()
            return
        load_model_state_dict(self.model, payload["model"], strict=strict)
        if self.optimizer is not None and payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        amp = payload.get("amp") or {}
        if self.scaler is not None and amp.get("scaler") is not None:
            self.scaler.load_state_dict(amp["scaler"])
        restore_rng_state(payload.get("rng") or {})
        if payload.get("sampler"):
            self.load_sampler_state(payload["sampler"])
        self.state.updates = int(payload.get("updates", 0))
        self.state.micro = int(payload.get("micro", self.state.updates * self.accum_steps))
        self.state.resume_updates = self.state.updates
        self.state.updates_this_resume = 0
        self._schedule_lr()

    # -- loop -------------------------------------------------------------------

    def _schedule_lr(self) -> None:
        apply_lr(
            self.optimizer,
            self.base_lrs,
            update=self.state.updates,
            updates=self.updates,
            warmup=self.warmup_updates,
            min_ratio=self.min_lr_ratio,
        )

    def _optimizer_step(self) -> dict[str, Any]:
        torch = _torch()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = None
        term_records: dict[str, dict[str, Any]] = {}
        batch_info: dict[str, Any] = {}
        conflict = bool(self.grad_conflict_every) and (
            (self.state.updates + 1) % self.grad_conflict_every == 0
        )
        conflict_payload: dict[str, Any] = {}
        for micro in range(self.accum_steps):
            batch = self.source.next_micro_batch()
            with autocast_context(self.amp, self.device):
                loss, terms, info = self.loss_fn(
                    self.model,
                    batch,
                    MicroInfo(update=self.state.updates, micro=micro, seed=self.seed),
                )
            if loss is None:
                raise TrainNotAvailable(
                    "no trainable loss term is available for this batch: "
                    + "; ".join(
                        f"{name}={record.get('reason') or record.get('status')}"
                        for name, record in terms.items()
                    )
                )
            term_tensors = {
                name: record.get("tensor")
                for name, record in terms.items()
                if torch.is_tensor(record.get("tensor"))
            }
            scaled = loss / float(self.accum_steps)
            if conflict and micro == self.accum_steps - 1 and len(term_tensors) >= 2:
                # autograd.grad keeps the graph so the backward below still works.
                conflict_payload = term_gradient_conflict(self.model, term_tensors)
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            for name, record in terms.items():
                record.pop("tensor", None)
                merged = term_records.setdefault(
                    name,
                    {
                        "status": record.get("status", "unavailable"),
                        "value": 0.0,
                        "sum": 0.0,
                        "activation_count": 0,
                        "micro_batches": 0,
                        "reason": record.get("reason", ""),
                        "kind": record.get("kind", ""),
                    },
                )
                if record.get("status") in ("ok", "detached"):
                    merged["value"] += float(record.get("value") or 0.0)
                    merged["sum"] += float(record.get("sum") or 0.0)
                    merged["activation_count"] += int(record.get("activation_count") or 0)
                    merged["micro_batches"] += 1
                elif merged.get("status") in ("ok", "detached"):
                    merged["status"] = record.get("status", "unavailable")
                    merged["reason"] = record.get("reason", "")
                for key in ("label_source", "score_kind", "gradient_available"):
                    if key in record:
                        merged[key] = record[key]
            for key, value in (info or {}).items():
                batch_info[key] = value
            total_loss = loss if total_loss is None else total_loss + loss
            self.state.micro += 1

        if self.clip_grad and self.clip_grad > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model_parameters(self.model), self.clip_grad)
        stats = gradient_statistics(self.model)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.state.updates += 1
        self.state.updates_this_resume += 1
        self._schedule_lr()
        record: dict[str, Any] = {
            "update": int(self.state.updates),
            "loss": None if total_loss is None else float(total_loss.detach().item()),
            "lr": [float(group["lr"]) for group in self.optimizer.param_groups],
            "terms": term_records,
            "time": _now(),
            **stats,
            **conflict_payload,
        }
        record.update({f"batch_{key}": value for key, value in batch_info.items()})
        if len(self.state.history) < self.history_limit:
            self.state.history.append(_json_ready(record))
        return record

    def train(
        self,
        *,
        until: int | None = None,
        log_every: int = 50,
        checkpoint_every: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_kwargs: Mapping[str, Any] | None = None,
        logger: Any = None,
        dev_eval: Any = None,
        dev_eval_every: int = 0,
        stop_after: int | None = None,
    ) -> dict[str, Any]:
        """Run to ``until`` updates (or ``stop_after`` more); return a summary."""
        target = int(self.updates if until is None else until)
        if stop_after is not None:
            target = min(target, self.state.updates + int(stop_after))
        summary: dict[str, Any] = {"updates_start": int(self.state.updates), "records": []}
        while self.state.updates < target:
            record = self._optimizer_step()
            if logger is not None and (
                self.state.updates % max(1, log_every) == 0 or self.state.updates == target
            ):
                logger(record)
            if checkpoint_every and checkpoint_dir and self.state.updates % checkpoint_every == 0:
                path = Path(checkpoint_dir) / f"update-{self.state.updates:07d}.pt"
                self.save(path, **dict(checkpoint_kwargs or {}))
                summary["records"].append({"checkpoint": str(path), "update": self.state.updates})
            if dev_eval is not None and dev_eval_every and self.state.updates % dev_eval_every == 0:
                payload = dev_eval(self, self.state.updates)
                if logger is not None:
                    logger({"update": self.state.updates, "dev_eval": payload})
                summary["records"].append({"dev_eval": payload, "update": self.state.updates})
        return summary


def _json_ready(value: Any) -> Any:
    """Convert tensors/arrays inside an instrumentation record to plain values."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:  # noqa: BLE001 - multi-element tensors fall through
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------


def load_partition_manifest(path: str | Path) -> Any:
    """Load a v2 partition manifest (frozen P0 artefact); never re-partition."""
    from owac.frameworks.n02_full import data as n02_data

    payload = read_json(path)
    if isinstance(payload, Mapping) and "manifest" in payload and "episodes" not in payload:
        payload = payload["manifest"]
    return n02_data.PartitionManifest.from_json(payload)


def load_anchor_table(path: str | Path) -> Any:
    from owac.frameworks.n02_full import data as n02_data

    payload = read_json(path)
    if isinstance(payload, Mapping) and "anchor_table" in payload and "entries" not in payload:
        payload = payload["anchor_table"]
    return n02_data.AnchorTable.from_json(payload)


class _AnchorWalker:
    """Shared deterministic walk over one anchor table's positions."""

    def __init__(
        self, dataset: Any, *, seed: int, platform: str, partition: str, salt: str
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.platform = str(platform)
        self.partition = str(partition)
        self.salt = str(salt)
        self._length = len(dataset)
        if self._length <= 0:
            raise TrainNotAvailable(f"{platform}/{partition} anchor table is empty")
        self._groups = self._episode_groups()
        self._order = self._permutation(0)
        self._cursor = 0
        self._seen_anchors: set[int] = set()
        self._seen_episodes: set[str] = set()
        self._frames_consumed = 0

    def _episode_groups(self) -> list[Any]:
        """Anchor indices grouped by source episode, in table order.

        Consecutive draws then read the same NPZ file instead of one file per frame;
        the group order is a pure function of the frozen anchor table.
        """
        groups: dict[str, list[int]] = {}
        for index, anchor in enumerate(self.dataset.anchor_table):
            groups.setdefault(str(anchor.relative_path), []).append(int(index))
        return [groups[key] for key in sorted(groups)]

    def _permutation(self, epoch: int) -> Any:
        """A flat order over anchor positions: shuffled episodes, ordered frames.

        The sampling unit stays the anchor; only the *order* changes, so the set of
        frames in an epoch and every coverage count are unchanged.
        """
        np = _numpy()
        rng = np.random.default_rng(
            _derive_seed(self.salt, self.partition, self.platform, self.seed, epoch)
        )
        episodes = rng.permutation(len(self._groups))
        return np.concatenate(
            [np.asarray(self._groups[int(index)], dtype=np.int64) for index in episodes]
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "partition": self.partition,
            "cursor": int(self._cursor),
            "seed": self.seed,
        }

    def coverage(self) -> dict[str, Any]:
        """Unique data actually drawn, with the dedup rule stated.

        Unique anchors are counted by their exact position in the frozen anchor
        table (a content-derived index, not a batch slot) and unique episodes by
        the anchor's ``relative_path``. Repeated draws of one anchor are counted as
        exposures, never as new data.
        """
        return {
            "partition": self.partition,
            "platform": self.platform,
            "anchor_table_entries": int(self._length),
            "unique_anchors_seen": len(self._seen_anchors),
            "unique_episodes_seen": len(self._seen_episodes),
            "frames_consumed": int(self._frames_consumed),
            "dedup_rule": (
                "unique anchors: exact anchor-table position set; unique episodes: anchor "
                "relative_path set; a repeated draw is an exposure, not new data"
            ),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("seed", self.seed)) != self.seed:
            raise TrainNotAvailable(
                "checkpoint sampler seed does not match the requested seed: "
                f"{payload.get('seed')} != {self.seed}"
            )
        self._cursor = int(payload.get("cursor", 0))
        self._order = self._permutation(self._cursor // max(1, self._length))

    def next_index(self) -> int:
        epoch, offset = divmod(self._cursor, self._length)
        slot = int(self._order[offset])
        self._cursor += 1
        self._seen_anchors.add(slot)
        try:
            self._seen_episodes.add(str(self.dataset.anchor_table[slot].relative_path))
        except (AttributeError, IndexError, TypeError):
            pass
        self._frames_consumed += 1
        if offset + 1 == self._length:
            self._order = self._permutation(epoch + 1)
        return slot


class RealFrameBatchSource(_AnchorWalker):
    """Batched single frames from one platform's frozen anchors.

    The walk order is a per-epoch permutation derived from
    ``(partition, platform, model seed)`` -- a data-side seed, never a model
    weight seed -- so a resume replays exactly the same frames from the recorded
    position.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        micro_batch_frames: int,
        seed: int,
        platform: str,
        partition: str = "train",
        threads: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            platform=platform,
            partition=partition,
            salt="n02-full-p1-pretrain-order",
        )
        self.micro_batch_frames = int(micro_batch_frames)
        if self.micro_batch_frames <= 0:
            raise ValueError("micro_batch_frames must be positive")
        self.threads = max(1, int(threads))
        self._pool = None

    def _executor(self) -> Any:
        if self.threads <= 1 or self.micro_batch_frames <= 1:
            return None
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=min(self.threads, self.micro_batch_frames))
        return self._pool

    def _payloads(self, slots: Sequence[int]) -> list[dict[str, Any]]:
        """One shared-contract frame per camera, built once per sample."""
        from owac.frameworks.n02_full import data as n02_data

        def _one(slot: int) -> dict[str, Any]:
            return n02_data.sample_frames(self.dataset[slot])

        pool = self._executor()
        # ThreadPoolExecutor.map preserves order, so the batch composition stays a
        # pure function of the sampler position whatever the scheduling does.
        return list(pool.map(_one, slots)) if pool is not None else [_one(slot) for slot in slots]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def next_micro_batch(self) -> dict[str, Any]:
        slots = [self.next_index() for _ in range(self.micro_batch_frames)]
        payloads = self._payloads(slots)
        camera_ids = tuple(payloads[0])
        frames: dict[str, Any] = {}
        for camera_id in camera_ids:
            rows = [payload[camera_id] for payload in payloads]
            frames[camera_id] = stack_measurement_frames(
                rows, camera_id=camera_id, squeeze_leading=True
            )
        return frames


class CursorFrameSource:
    """A fixed cursor over a dataset that never shuffles (dev evaluation)."""

    def __init__(self, dataset: Any, *, micro_batch_frames: int, limit: int) -> None:
        self.dataset = dataset
        self.micro_batch_frames = int(micro_batch_frames)
        self.limit = int(limit)
        self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"cursor": int(self._cursor)}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._cursor = int(payload.get("cursor", 0))

    def next_micro_batch(self) -> dict[str, Any]:
        from owac.frameworks.n02_full import data as n02_data

        count = min(self.micro_batch_frames, max(0, self.limit - self._cursor))
        if count <= 0:
            self._cursor = 0
            count = min(self.micro_batch_frames, self.limit)
        payloads = [
            n02_data.sample_frames(self.dataset[self._cursor + offset]) for offset in range(count)
        ]
        self._cursor += count
        frames: dict[str, Any] = {}
        for camera_id in tuple(payloads[0]):
            rows = [payload[camera_id] for payload in payloads]
            frames[camera_id] = stack_measurement_frames(
                rows, camera_id=camera_id, squeeze_leading=True
            )
        return frames


class SyntheticFrameBatchSource:
    """Batched CFA4 frames from the controlled generator, for the stem pretraining.

    The raw3 cache cannot feed a CFA4 stem (T1 never converts the modes), so the
    stem is pretrained on the generator's declared CFA4 measurements. Frames are
    taken in generator order and stacked to ``(B, C, H, W)``; no truth enters this
    path, it is masked reconstruction only.
    """

    def __init__(
        self,
        spec: MeasurementSpec,
        *,
        micro_batch_frames: int,
        seed: int,
        canvas: tuple[int, int] = (64, 64),
    ) -> None:
        self.spec = spec
        self.micro_batch_frames = int(micro_batch_frames)
        self.seed = int(seed)
        self.canvas = tuple(canvas)
        self._dataset = None
        self._iterator = None

    def _next_sequence(self) -> Any:
        from owac.frameworks.n02_full import synthetic as synth

        if self._dataset is None:
            self._dataset = synth.generate_dataset(
                sensors=_synthetic_sensors(self.spec.synthetic_sensors),
                measurement_model=synth.MeasurementModel(**dict(self.spec.measurement_model)),
                partitions=("train",),
                scenes=max(32, int(self.spec.synthetic_scenes)),
                frames=max(2, int(self.spec.frames_per_episode)),
                views_per_scene=1,
                seed=self.seed,
                canvas=self.canvas,
                action_horizon=int(self.spec.action_horizon),
            )
            self._iterator = iter(self._dataset)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._dataset)
            return next(self._iterator)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "synthetic-frames",
            "seed": self.seed,
            "scenes": None if self._dataset is None else int(self._dataset.counters.sequences),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        return None

    def next_micro_batch(self) -> dict[str, Any]:
        rows: dict[str, list[Any]] = {}
        collected = 0
        while collected < self.micro_batch_frames:
            sequence = self._next_sequence()
            bundles = [item.bundle for item in sequence.inference]
            for bundle in bundles:
                for camera_id, frame in bundle.frames.items():
                    rows.setdefault(str(camera_id), []).append(frame)
                collected += 1
                if collected >= self.micro_batch_frames:
                    break
            if collected >= self.micro_batch_frames:
                break
        return {
            camera_id: stack_measurement_frames(frames, camera_id=camera_id, squeeze_leading=True)
            for camera_id, frames in rows.items()
        }

    def coverage(self) -> dict[str, Any]:
        sequences = 0 if self._dataset is None else int(self._dataset.counters.sequences)
        return {
            "partition": "synthetic-train",
            "platform": "synthetic",
            "anchor_table_entries": None,
            "unique_anchors_seen": sequences,
            "unique_episodes_seen": sequences,
            "frames_consumed": None,
            "dedup_rule": "generator sequences; each sequence is an independent controlled scene",
        }


class RoundRobinFrames:
    """Alternate micro-batches between platforms so resolutions never mix."""

    def __init__(self, sources: Mapping[str, Any], *, micro_batch_frames: int) -> None:
        if not sources:
            raise TrainNotAvailable("no frame source is available")
        self.sources = dict(sources)
        self.micro_batch_frames = int(micro_batch_frames)
        self._names = sorted(self.sources)
        self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": int(self._cursor),
            "platforms": {name: source.state_dict() for name, source in self.sources.items()},
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._cursor = int(payload.get("cursor", 0))
        for name, state in (payload.get("platforms") or {}).items():
            if name in self.sources:
                self.sources[name].load_state_dict(state)

    def next_micro_batch(self) -> dict[str, Any]:
        name = self._names[self._cursor % len(self._names)]
        self._cursor += 1
        return self.sources[name].next_micro_batch()


def _load_normalisation(data_spec: DataSpec) -> Any:
    if not data_spec.stats_path:
        return None
    from owac.frameworks.n02_full import data as n02_data

    path = Path(data_spec.stats_path)
    if not path.is_file():
        return None
    try:
        return n02_data.load_formal_normalisation(data_spec.dataset_format, path)
    except Exception:  # noqa: BLE001 - an unusable stats file is recorded as absent
        return None


def open_anchor_dataset(data_spec: DataSpec, partition: str) -> Any:
    """Open one frozen-anchor ``ManifestWindowDataset`` for a declared partition."""
    from owac.frameworks.n02_full import data as n02_data

    manifest_path = data_spec.manifest
    manifest = load_partition_manifest(manifest_path)
    anchor_path = data_spec.anchor_path(partition)
    if not anchor_path:
        raise TrainNotAvailable(
            f"training.data anchors.{partition} is not declared", missing=(anchor_path,)
        )
    anchors = load_anchor_table(anchor_path)
    cache_root = data_spec.cache_root or manifest.cache_root
    root = Path(cache_root)
    if not root.is_dir():
        raise TrainNotAvailable(
            f"the declared data root is not present on this host: {root}",
            missing=(str(root),),
        )
    window = (
        n02_data.WindowOptions(**data_spec.window_options) if data_spec.window_options else None
    )
    normalisation = _load_normalisation(data_spec)
    stats_path = Path(data_spec.stats_path) if data_spec.stats_path else None
    dataset = n02_data.ManifestWindowDataset(
        cache_root,
        manifest,
        partition,
        anchors,
        window_options=window,
        normalisation=normalisation,
    )
    # Provenance of the declared inputs, recorded in the run manifest: which
    # frozen manifest/anchors and which formal normalisation statistics the run
    # actually consumed (a missing stats file is reported as not loaded).
    dataset.n02_provenance = {
        "cache_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_declared_hash": data_spec.manifest_hash,
        "manifest_content_hash": manifest.content_hash(),
        "anchor_path": str(anchor_path),
        "anchor_declared_hash": data_spec.anchor_sha256(partition),
        "anchor_content_hash": anchors.sha256(),
        "anchors": len(anchors),
        "window": _window_payload(dataset.window_options),
        "stats_path": str(stats_path) if stats_path else "",
        "stats_sha256": _sha256_file(stats_path) if stats_path else None,
        "stats_declared": bool(data_spec.stats_path),
        "normalisation_loaded": normalisation is not None,
        "formal_normalisation": data_spec.formal_normalisation,
        "dataset_format": data_spec.dataset_format,
    }
    return dataset


def data_specs(cfg: n02_config.N02FullConfig) -> dict[str, DataSpec]:
    """Resolve every declared platform section of ``training.data``."""
    section = cfg.training.get("data", {}) if isinstance(cfg.training, dict) else {}
    if not isinstance(section, Mapping):
        raise TrainingSpecError("training.data: expected an object")
    specs: dict[str, DataSpec] = {}
    for platform, entry in section.items():
        specs[str(platform)] = _section(DataSpec, entry, f"training.data.{platform}")
    return specs


def _window_payload(window: Any) -> Any:
    """Serialise one window declaration whatever shape the P0 dataclass has."""
    if window is None:
        return {}
    if hasattr(window, "as_dict"):
        return window.as_dict()
    if dataclasses.is_dataclass(window):
        return dataclasses.asdict(window)
    return str(window)


def coverage_report(
    sources: Mapping[str, Any],
    cfg: n02_config.N02FullConfig,
    spec: PretrainSpec,
) -> dict[str, Any]:
    """Independent data coverage and equivalent exposures of one run (plan 8.1).

    ``equivalent_exposures`` is the consumed frame count divided by the number of
    available training anchors in the declared partitions: the plan asks for at
    least 20, and a run below it says so instead of implying full coverage. Augmented
    views are never counted as new data (the walker only draws real anchors).
    """
    per_platform: dict[str, Any] = {}
    frames = 0
    available = 0
    for name, source in sorted(sources.items()):
        reporter = getattr(source, "coverage", None)
        if not callable(reporter):
            continue
        entry = reporter()
        per_platform[name] = entry
        frames += int(entry.get("frames_consumed", 0))
        available += int(entry.get("anchor_table_entries", 0))
    exposures = (frames / available) if available else None
    partitions: dict[str, Any] = {}
    for platform, data_spec in data_specs(cfg).items():
        partitions[platform] = {
            partition: {
                "path": data_spec.anchor_path(partition),
                "declared_sha256": data_spec.anchor_sha256(partition),
                "declared_anchors": (
                    data_spec.anchors.get(partition, {}).get("anchors")
                    if isinstance(data_spec.anchors.get(partition), Mapping)
                    else None
                ),
            }
            for partition in ("train", "dev", "calibration")
        }
        partitions[platform]["manifest_path"] = data_spec.manifest
        partitions[platform]["manifest_declared_hash"] = data_spec.manifest_hash
    return {
        "per_platform": per_platform,
        "frames_consumed_total": frames,
        "anchors_available_total": available,
        "equivalent_exposures": exposures,
        "plan_min_equivalent_exposures": 20,
        "meets_plan_8_1": None if exposures is None else bool(exposures >= 20),
        "partition_hashes": partitions,
        "micro_batch_frames": int(spec.micro_batch_frames),
        "effective_batch_frames": int(spec.effective_batch_frames),
    }


def partition_fingerprint(datasets: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen partition/anchor hash recorded in checkpoints and manifests."""
    payload: dict[str, Any] = {}
    if not datasets:
        payload["sources"] = "injected"
        payload["partition_hash"] = _sha256_text(json.dumps(payload, sort_keys=True))
        return payload
    for name, dataset in sorted(datasets.items()):
        payload[name] = {
            "partition": str(getattr(dataset, "partition", "")),
            "manifest_hash": dataset.manifest.content_hash(),
            "anchor_hash": dataset.anchor_table.sha256(),
            "anchors": len(dataset),
            "window": _window_payload(dataset.window_options),
        }
        provenance = getattr(dataset, "n02_provenance", None)
        if provenance:
            payload[name].update(provenance)
        declared = getattr(dataset, "declared_hashes", None)
        if declared:
            payload[name]["declared_hashes"] = dict(declared)
    payload["partition_hash"] = _sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return payload


# ---------------------------------------------------------------------------
# pretraining
# ---------------------------------------------------------------------------

OBSERVATION_MODULE = "owac.frameworks.n02_full.observation"


def model_section(cfg: n02_config.N02FullConfig) -> dict:
    """The ``training.model`` declaration (candidate count, modes, decoder widths)."""
    training = cfg.training if isinstance(cfg.training, dict) else {}
    section = training.get("model", {})
    if not isinstance(section, Mapping):
        raise TrainingSpecError("training.model: expected an object")
    return dict(section)


def p1_component_config(cfg: n02_config.N02FullConfig, *, world_size: int | None = None) -> dict:
    """Config view the P1 components read, with every key resolved explicitly.

    ``factors.ObservationModel`` reads ``candidate_world_size``/``feature_dim``/... from
    its config argument; the frozen contract fixes the starting world count at 8, so
    the adapter takes N02FullConfig.worlds unless ``training.model`` overrides it.
    """
    section = model_section(cfg)
    # ``candidate_world_size`` is the number of *objects* a candidate geometry
    # places (it must equal CandidateDictionary.world_size()), not the number of
    # candidates; a controlled run pins it to the probed scene.
    payload = {
        "candidate_world_size": int(
            world_size
            if world_size is not None
            else section.get("candidate_world_size", cfg.worlds)
        ),
        "feature_dim": int(section.get("feature_dim", cfg.slot_dim)),
        "candidate_embedding_dim": int(section.get("candidate_embedding_dim", 32)),
        "decoder_grid": int(section.get("decoder_grid", 8)),
        "decoder_channels": int(section.get("decoder_channels", 32)),
        "decoder_hidden": int(section.get("decoder_hidden", 64)),
        "structure_version": str(section.get("structure_version", "p1-fixed-entities")),
        "initial_prior_version": str(section.get("initial_prior_version", "")),
        "input_mode": str(section.get("input_mode", "raw3")),
    }
    shapes = section.get("sensor_shape") or section.get("camera_shapes")
    if isinstance(shapes, Mapping):
        payload["sensor_shape"] = {str(key): tuple(value) for key, value in shapes.items()}
    return payload


def _declared_kwargs(cls: Any, wanted: Mapping[str, Any], where: str) -> dict:
    """Pass only the keyword arguments the constructor actually declares."""
    import inspect

    parameters = set(inspect.signature(cls.__init__).parameters)
    unknown = sorted(set(wanted) - parameters - {"self"})
    if unknown:
        raise TrainNotAvailable(
            f"{where} does not accept the declared key(s): {', '.join(unknown)}",
            missing=tuple(unknown),
        )
    return {key: value for key, value in wanted.items() if key in parameters}


def build_pretrainer(
    cfg: n02_config.N02FullConfig, spec: PretrainSpec, device: str, *, seed: int
) -> Any:
    """Instantiate T1's FCMAE pretrainer; a missing module is ``not_run``."""
    del seed
    try:
        module = importlib.import_module(OBSERVATION_MODULE)
    except ImportError as exc:
        raise TrainNotAvailable(
            f"{OBSERVATION_MODULE} is not available: {type(exc).__name__}: {exc}",
            missing=(OBSERVATION_MODULE,),
        ) from exc
    section = model_section(cfg)
    patch_size = int(section.get("patch_size", 32))
    if patch_size != 32:
        raise TrainingSpecError(
            f"training.model.patch_size: the frozen P1 value is 32 (it must match the "
            f"four-stage backbone stride), got {patch_size}"
        )
    wanted = {
        "input_mode": str(section.get("input_mode", "raw3")),
        "in_chans": section.get("in_chans"),
        "depths": tuple(cfg.depths),
        "dims": tuple(cfg.dims),
        "patch_size": patch_size,
        "mask_ratio": float(spec.mask_ratio),
        "decoder_embed_dim": int(section.get("decoder_embed_dim", cfg.slot_dim)),
        "decoder_depth": int(section.get("decoder_depth", 1)),
        "norm_pix_loss": bool(section.get("norm_pix_loss", False)),
        "img_size": (
            int(section.get("img_height", cfg.input_size)),
            int(section.get("img_width", cfg.input_size)),
        ),
        "alignment": 32,
    }
    cls = getattr(module, "RawFCMAEPretrainer", None)
    if cls is None:
        raise TrainNotAvailable(
            f"{OBSERVATION_MODULE} exposes no RawFCMAEPretrainer",
            missing=(f"{OBSERVATION_MODULE}.RawFCMAEPretrainer",),
        )
    return cls(**_declared_kwargs(cls, wanted, "RawFCMAEPretrainer")).to(device)


def camera_groups(batch: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Group cameras that share a channel count and grid (T1 needs one layout per call)."""
    groups: dict[tuple, list[str]] = {}
    np = _numpy()
    for camera_id, frame in batch.items():
        shape = tuple(np.asarray(frame.values).shape[1:])
        groups.setdefault(shape, []).append(str(camera_id))
    return [tuple(cameras) for _, cameras in sorted(groups.items(), key=lambda item: str(item[0]))]


def pretrain_loss_fn(model: Any, batch: Any, info: MicroInfo) -> tuple[Any, dict, dict]:
    """One FCMAE micro-batch: the module's own masked loss, reported verbatim.

    Cameras with different layouts are forwarded as separate groups (T1 requires a
    shared channel count and padded grid per call); the reported value is the mean
    over groups and the activation count is their sum, so a mixed-layout batch is
    never silently truncated.
    """
    torch = _torch()
    groups = camera_groups(batch)
    if not groups:
        raise TrainNotAvailable("empty frame batch")
    losses: list[Any] = []
    activations = 0
    saturated = 0
    for cameras in groups:
        output = model(batch, camera_ids=cameras)
        loss = getattr(output, "loss", None)
        if loss is None:
            raise TrainNotAvailable("FCMAE output has no 'loss' attribute")
        losses.append(loss)
        valid = getattr(output, "valid_target_count", None)
        sat = getattr(output, "saturated_count", None)
        activations += int(valid.item()) if torch.is_tensor(valid) else int(valid or 0)
        saturated += int(sat.item()) if torch.is_tensor(sat) else int(sat or 0)
    loss = losses[0] if len(losses) == 1 else torch.stack(losses).mean()
    terms = {
        "masked_reconstruction": {
            "status": "ok",
            "value": float(loss.detach().item()),
            "sum": float(loss.detach().item()) * activations,
            "activation_count": activations,
            "reason": "",
            "kind": "fcmae_masked_reconstruction",
            "saturated_count": saturated,
            "camera_groups": [list(group) for group in groups],
        }
    }
    return loss, terms, {"camera_ids": list(batch), "valid_target_count": activations}


def make_pretrain_dev_eval(
    dev_source: Any, *, device: str, micro_batch_frames: int, dev_frames: int = 256
) -> Any:
    """Fixed dev reconstruction evaluation: no grad, dropout off, fixed frames.

    The number of micro-batches is derived from the declared dev frame budget and
    the micro-batch size, so the evaluation is a fixed, finite subset -- a dev
    source that never ends (the resumable walker) cannot turn it into an
    unbounded loop.
    """
    micro_batches = max(1, math.ceil(int(dev_frames) / max(1, int(micro_batch_frames))))

    def _dev_eval(trainer: P1Trainer, update: int) -> dict[str, Any]:
        torch = _torch()
        # The dev subset is fixed: every evaluation restarts the declared dev walk,
        # so a curve compares the same frames instead of a rolling window.
        reset = getattr(dev_source, "load_state_dict", None)
        if callable(reset):
            reset({"cursor": 0})
        model = trainer.model
        was_training = model.training
        model.eval()
        losses = []
        activations = 0
        detail: dict[str, Any] = {"dev_local_detail": None, "dev_local_detail_reason": ""}
        try:
            with torch.no_grad():
                for _ in range(micro_batches):
                    try:
                        batch = dev_source.next_micro_batch()
                    except StopIteration:
                        break
                    if not batch:
                        break
                    output = model(batch, camera_ids=tuple(batch))
                    loss = getattr(output, "loss", None)
                    if loss is None:
                        break
                    losses.append(float(loss.detach().item()))
                    valid = getattr(output, "valid_target_count", None)
                    activations += int(valid.item()) if torch.is_tensor(valid) else int(valid or 0)
                    if detail["dev_local_detail"] is None and not detail["dev_local_detail_reason"]:
                        detail.update(_local_detail(output, batch))
        finally:
            if was_training:
                model.train()
        if not losses:
            return {"status": "not_run", "reason": "dev source produced no batch"}
        payload = {
            "status": "ok",
            "update": int(update),
            "dev_masked_reconstruction_loss": sum(losses) / len(losses),
            "dev_micro_batches": len(losses),
            "dev_valid_target_count": activations,
            "frames_per_micro_batch": int(micro_batch_frames),
            "dev_frame_budget": int(dev_frames),
        }
        payload.update(detail)
        return payload

    return _dev_eval


def _local_detail(output: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Local-detail ratio of the masked reconstruction (a declared dev diagnostic).

    It compares the mean absolute reconstruction error on the highest-gradient
    decile of the target frame with the mean absolute error over the whole frame:
    a value near 1 means error is spread out, a larger value means the error sits
    on structure. It is reported only when the reconstruction has the frame shape;
    a padded/mismatched layout is reported as unavailable with the reason, never
    computed on a guessed alignment.
    """
    np = _numpy()
    reconstruction = getattr(output, "reconstruction", None)
    if reconstruction is None:
        return {"dev_local_detail": None, "dev_local_detail_reason": "no reconstruction"}
    tensor = reconstruction.detach().cpu().numpy() if hasattr(reconstruction, "detach") else None
    if tensor is None:
        return {
            "dev_local_detail": None,
            "dev_local_detail_reason": "reconstruction is not a tensor",
        }
    if tensor.ndim == 5:  # camera axis first
        tensor = tensor[0]
    camera_id = next(iter(batch))
    target = np.asarray(batch[camera_id].values)
    if target.ndim == 5:
        target = target[:, 0]
    if tensor.shape != target.shape:
        return {
            "dev_local_detail": None,
            "dev_local_detail_reason": (
                f"reconstruction {tensor.shape} does not match the frame {target.shape} "
                "(padding/crop alignment); not computed on a guessed layout"
            ),
        }
    error = np.abs(tensor - target)
    gradient = np.zeros(target.shape)
    gradient[..., :-1, :] += np.abs(np.diff(target, axis=-2))
    gradient[..., :, :-1] += np.abs(np.diff(target, axis=-1))
    flat_gradient = gradient.reshape(-1)
    if flat_gradient.size == 0 or not np.any(flat_gradient > 0.0):
        return {
            "dev_local_detail": None,
            "dev_local_detail_reason": "the frame has no measurable spatial gradient",
        }
    cutoff = np.quantile(flat_gradient, 0.9)
    structure = flat_gradient >= cutoff
    flat_error = error.reshape(-1)
    overall = float(flat_error.mean())
    if overall <= 0.0:
        return {
            "dev_local_detail": None,
            "dev_local_detail_reason": "the mean absolute error is zero",
        }
    return {
        "dev_local_detail": float(flat_error[structure].mean()) / overall,
        "dev_local_detail_reason": "",
        "dev_local_detail_definition": (
            "mean|error| on the top-decile gradient pixels / mean|error|"
        ),
    }


def run_precision_check(
    cfg: n02_config.N02FullConfig,
    spec: PretrainSpec,
    context: RunContext,
    *,
    args: argparse.Namespace,
    device: str,
    platform: str,
    micro_batch: int | None = None,
    tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Error check of the declared reduced-precision mode against the FP32 reference.

    One fixed batch (fixed seed, fixed anchors) is forwarded, differentiated and
    stepped from identical initial parameters under each mode; the receipt reports
    max-abs and max-rel differences of the loss, the full gradient vector and the
    parameter step. FP32 is the reference: a true FP64 reference is not reachable
    for this 92M-parameter ConvNeXt in a bounded run, which the receipt states.
    """
    torch = _torch()
    tolerances = dict(tolerances or {"tf32": 1e-2, "bf16": 5e-2})
    if isinstance(spec, MeasurementSpec):
        # The measurement arm has the log-variance and quantisation terms, so it needs
        # its own check rather than inheriting the pretraining verdict.
        sources = build_measurement_sources(cfg, spec, context.seed)
        probed = sources.probe_world_size() if hasattr(sources, "probe_world_size") else None
        batch = sources.next_micro_batch()
        loss_fn = measurement_loss_fn(
            None, spec, sensor_models=build_measurement_sensor_models(cfg, spec)
        )

        def _factory() -> Any:
            return build_observation_model(
                cfg,
                device,
                units=spec.units or _synthetic_units(spec),
                world_size=probed,
            )

        reference_state = None
        results: dict[str, dict[str, Any]] = {}
        for mode in ("off", "tf32", "bf16"):
            model = _factory()
            if reference_state is None:
                reference_state = {
                    key: value.detach().clone()
                    for key, value in flatten_state_dict(model_state_dict(model)).items()
                }
            else:
                load_model_state_dict(model, _nested_state(model, reference_state))
            precision = configure_precision(mode, device)
            optimizer = torch.optim.AdamW(
                [p for p in model_parameters(model) if p.requires_grad],
                lr=1e-5,
                betas=(0.9, 0.95),
            )
            with autocast_context(mode, device):
                loss, _terms, info = loss_fn(model, batch, MicroInfo(0, 0, context.seed))
            if loss is None:
                raise TrainNotAvailable("the measurement batch produced no trainable loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            vector = _grad_vector([p for p in model_parameters(model) if p.requires_grad])
            if vector is None:
                raise TrainNotAvailable(f"mode {mode!r} produced no gradient")
            torch.nn.utils.clip_grad_norm_(model_parameters(model), spec.grad_clip)
            optimizer.step()
            step_parts = []
            for key, value in flatten_state_dict(model_state_dict(model)).items():
                initial = reference_state.get(key)
                if initial is None:
                    continue
                step_parts.append(
                    (value.detach().to(torch.float32) - initial.to(torch.float32)).reshape(-1)
                )
            step = torch.cat(step_parts) if step_parts else torch.zeros(1)
            results[mode] = {
                "loss": float(loss.detach().item()),
                "grad_norm": float(vector.norm().item()),
                "step_norm": float(step.norm().item()),
                "valid_target_count": int(info.get("valid_target_count", 0) or 0),
                "precision": precision,
                "grad_slice": vector[:256].detach().to(torch.float32).cpu(),
                "step_slice": step[:256].detach().to(torch.float32).cpu(),
            }
        return _finish_precision_receipt(context, spec, platform, micro_batch, tolerances, results)

    specs = data_specs(cfg)
    if platform not in specs:
        raise TrainNotAvailable(f"training.data.{platform} is not declared")
    dataset = open_anchor_dataset(specs[platform], "train")
    source = RealFrameBatchSource(
        dataset,
        micro_batch_frames=int(micro_batch or spec.micro_batch_frames),
        seed=context.seed,
        platform=platform,
        threads=spec.dataloader_threads,
    )
    batch = source.next_micro_batch()
    source.close()
    reference_state: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {}
    for mode in ("off", "tf32", "bf16"):
        model = build_pretrainer(cfg, spec, device, seed=context.seed)
        if reference_state is None:
            reference_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            model.load_state_dict(reference_state)
        precision = configure_precision(mode, device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-5, betas=(0.9, 0.95)
        )
        with autocast_context(mode, device):
            loss, _terms, info = pretrain_loss_fn(model, batch, MicroInfo(0, 0, context.seed))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        vector = _grad_vector([p for p in model.parameters() if p.requires_grad])
        if vector is None:
            raise TrainNotAvailable(f"mode {mode!r} produced no gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        optimizer.step()
        step_parts = []
        for key, value in model.state_dict().items():
            initial = reference_state.get(key)
            if initial is None:
                continue
            step_parts.append(
                (value.detach().to(torch.float32) - initial.to(torch.float32)).reshape(-1)
            )
        step = torch.cat(step_parts) if step_parts else torch.zeros(1)
        results[mode] = {
            "loss": float(loss.detach().item()),
            "grad_norm": float(vector.norm().item()),
            "step_norm": float(step.norm().item()),
            "valid_target_count": int(info.get("valid_target_count", 0)),
            "precision": precision,
            "grad_slice": vector[:256].detach().to(torch.float32).cpu(),
            "step_slice": step[:256].detach().to(torch.float32).cpu(),
        }
    return _finish_precision_receipt(
        context, spec, platform, micro_batch, tolerances, results, is_measurement=False
    )


def bound_torch_threads(default: int = 8) -> int:
    """Cap torch's intra-op threads for the small-tensor paths.

    A ratio-arm update is tiny-tensor work; at the host's full thread count the ops thrash
    (the evaluation path measured 880 CPU-s per step versus 1.9 at four threads). The cap is
    a scheduling change only and can be overridden with ``OWAC_TRAIN_THREADS``.
    """
    import os

    try:
        torch = _torch()
        threads = int(os.environ.get("OWAC_TRAIN_THREADS", str(default)))
        if threads > 0:
            torch.set_num_threads(threads)
        return threads
    except Exception:  # noqa: BLE001 - never break a run for a scheduling tweak
        return 0


def budget_start(resume: Mapping[str, Any] | None, stage: str) -> int:
    """The update counter a run starts from, ignoring a *cross-stage* Base inheritance.

    ``--updates N`` means N updates of *this* stage. A measurement run that inherits its
    Base from a pretraining checkpoint starts its own counter at zero (the pretraining
    updates are a different stage's budget and stay on the receipt as ``source_updates``);
    counting them would silently multiply the requested budget.
    """
    if not resume:
        return 0
    source_stage = str(resume.get("stage", ""))
    if source_stage and source_stage != stage:
        return 0
    return int(resume.get("updates", 0))


def save_final_checkpoint(trainer: Any, path: Path, **kwargs: Any) -> dict[str, Any]:
    """Write the final checkpoint, reporting a failure instead of dying silently.

    A host- or container-level kill still cannot be caught, but an application-level
    failure (disk, serialisation, permissions) now prints its traceback into the run log
    and leaves a structured record, so a later diagnosis never faces an empty log.
    """
    import traceback

    torch = _torch()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the receipt must survive any save failure
        traceback.print_exc()
        return {
            "status": "fail",
            "path": str(path),
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    size = path.stat().st_size if path.exists() else 0
    payload: dict[str, Any] = {"status": "ok", "path": str(path), "bytes": int(size)}
    try:
        payload["cuda_peak_allocated_mb"] = round(
            float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 1
        )
    except Exception:  # noqa: BLE001 - resource figures are evidence, never a requirement
        pass
    try:
        import resource

        payload["rss_peak_mb"] = round(
            float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0, 1
        )
    except Exception:  # noqa: BLE001
        pass
    print(f"[final-checkpoint] {payload}", flush=True)
    return payload


def _finish_precision_receipt(
    context: RunContext,
    spec: Any,
    platform: str,
    micro_batch: int | None,
    tolerances: Mapping[str, float],
    results: Mapping[str, Mapping[str, Any]],
    *,
    is_measurement: bool = True,
) -> dict[str, Any]:
    """Compare each reduced-precision mode against the FP32 reference."""
    reference = results["off"]
    receipt: dict[str, Any] = {
        "stage": context.stage,
        "measurement_arm": bool(is_measurement),
        "spec": (
            {
                "ce_temperature": getattr(spec, "ce_temperature", None),
                "loss_weights": dict(getattr(spec, "loss_weights", {}) or {}),
                "frames_per_episode": getattr(spec, "frames_per_episode", None),
            }
            if is_measurement
            else {"mask_ratio": getattr(spec, "mask_ratio", None)}
        ),
        "platform": platform,
        "micro_batch_frames": int(
            micro_batch
            or getattr(spec, "micro_batch_frames", 0)
            or getattr(spec, "micro_batch_episodes", 0)
        ),
        "seed": context.seed,
        "reference": "fp32 (matmul TF32 and cuDNN TF32 explicitly disabled)",
        "fp64_reference": "not_run: a 92M-parameter ConvNeXt in FP64 is outside this run's bound",
        "tolerances": tolerances,
        "modes": {},
    }
    for mode in ("tf32", "bf16"):
        entry = results[mode]
        grad_slice = entry["grad_slice"]
        step_slice = entry["step_slice"]
        reference_grad_slice = reference["grad_slice"]
        reference_step_slice = reference["step_slice"]
        loss_abs = abs(entry["loss"] - reference["loss"])
        loss_rel = loss_abs / max(abs(reference["loss"]), 1e-12)
        grad_abs = float((grad_slice - reference_grad_slice).abs().max().item())
        grad_denominator = float(reference_grad_slice.abs().max().item()) or 1e-12
        step_abs = float((step_slice - reference_step_slice).abs().max().item())
        step_denominator = float(reference_step_slice.abs().max().item()) or 1e-12
        norm_rel = abs(entry["grad_norm"] - reference["grad_norm"]) / max(
            reference["grad_norm"], 1e-12
        )
        tolerance = float(tolerances.get(mode, 5e-2))
        candidates = (
            loss_rel,
            norm_rel,
            grad_abs / grad_denominator,
            entry["loss"],
            entry["grad_norm"],
            entry["step_norm"],
            reference["loss"],
        )
        # A non-finite difference is a failure, never a silent pass: NaN compares
        # False against every tolerance and would otherwise fall through.
        non_finite = not all(math.isfinite(float(value)) for value in candidates)
        entry_receipt = {
            "loss": entry["loss"],
            "grad_norm": entry["grad_norm"],
            "step_norm": entry["step_norm"],
            "loss_abs_diff": loss_abs,
            "loss_rel_diff": loss_rel,
            "grad_slice_max_abs_diff": grad_abs,
            "grad_slice_max_rel_diff": grad_abs / grad_denominator,
            "grad_norm_rel_diff": norm_rel,
            "step_slice_max_abs_diff": step_abs,
            "step_slice_max_rel_diff": step_abs / step_denominator,
            "tolerance": tolerance,
            "non_finite": non_finite,
            "verdict": (
                "fail"
                if non_finite
                else "pass"
                if max(loss_rel, norm_rel, grad_abs / grad_denominator) <= tolerance
                else "fail"
            ),
            "precision": entry["precision"],
        }
        receipt["modes"][mode] = entry_receipt
    receipt["status"] = (
        "ok" if all(entry["verdict"] == "pass" for entry in receipt["modes"].values()) else "fail"
    )
    return receipt


def run_pretrain(
    cfg: n02_config.N02FullConfig,
    spec: PretrainSpec,
    context: RunContext,
    *,
    lr_scale: float = 1.0,
    updates: int | None = None,
    device: str = "cpu",
    resume: Mapping[str, Any] | None = None,
    model_factory: Any = None,
    frame_sources: Mapping[str, Any] | None = None,
    dev_source: Any = None,
    checkpoint_spec: CheckpointSpec | None = None,
    synthetic_frames: bool = False,
) -> dict[str, Any]:
    """Driving loop of ``--stage p1-pretrain``."""
    torch = _torch()
    additional = updates is not None
    start = budget_start(resume, "p1-pretrain")
    total_updates = (start + int(updates)) if additional else int(spec.updates)
    if total_updates <= start:
        raise TrainNotAvailable(
            f"nothing to train: resume at {start} updates and budget {total_updates}"
        )
    peak_lr = float(spec.lr) * float(lr_scale)
    warmup = spec.warmup_updates(total_updates)
    if model_factory is None:

        def model_factory() -> Any:
            return build_pretrainer(cfg, spec, device, seed=context.seed)

    # The upstream module is checked before the data root: a missing module is
    # not_run, while a missing/unreadable dataset is a real failure.
    model = model_factory().to(device)

    datasets: dict[str, Any] = {}
    if frame_sources is None and synthetic_frames:
        # A CFA4 stem cannot be pretrained on the raw3 cache (T1 never converts the
        # modes), so it trains on the generator's declared CFA4 frames instead.
        measurement_spec = _section(
            MeasurementSpec, cfg.training.get("measurement", {}), "training.measurement"
        )
        frame_sources = {
            "synthetic": SyntheticFrameBatchSource(
                measurement_spec,
                micro_batch_frames=spec.micro_batch_frames,
                seed=context.seed,
                canvas=tuple(measurement_spec.canvas),
            )
        }
    if frame_sources is None:
        frame_sources = {}
        for platform, data_spec in data_specs(cfg).items():
            if platform not in spec.platforms:
                continue
            dataset = open_anchor_dataset(data_spec, "train")
            datasets[platform] = dataset
            frame_sources[platform] = RealFrameBatchSource(
                dataset,
                micro_batch_frames=spec.micro_batch_frames,
                seed=context.seed,
                platform=platform,
                threads=spec.dataloader_threads,
            )
    else:
        datasets = {
            name: source.dataset
            for name, source in frame_sources.items()
            if getattr(source, "dataset", None) is not None
        }
    if not frame_sources and not datasets:
        raise TrainNotAvailable("no pretraining frame source is available")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise TrainNotAvailable("the pretrainer has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=peak_lr, betas=tuple(spec.betas), weight_decay=spec.weight_decay
    )
    scaler = None
    if spec.amp == "bf16":
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    source = RoundRobinFrames(frame_sources, micro_batch_frames=spec.micro_batch_frames)
    trainer = P1Trainer(
        stage=context.stage,
        seed=context.seed,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        source=source,
        loss_fn=pretrain_loss_fn,
        updates=total_updates,
        accum_steps=spec.accum_steps,
        base_lrs=[peak_lr],
        warmup_updates=warmup,
        min_lr_ratio=spec.min_lr_ratio,
        clip_grad=spec.grad_clip,
        device=device,
        amp=spec.amp,
    )
    if resume is not None:
        trainer.resume_from(resume)
    checkpoint_spec = checkpoint_spec or CheckpointSpec()
    partition = partition_fingerprint(datasets)

    def _checkpoint_kwargs() -> dict[str, Any]:
        return {
            "partition": partition,
            "config_hash": context.config_hash,
            "run_id": context.run_id,
            "extras": {"lr_scale": float(lr_scale), "mask_ratio": float(spec.mask_ratio)},
        }

    dev_eval_unavailable = False
    if dev_source is None and synthetic_frames:
        # The real dev partition is raw3; a CFA4 stem cannot be evaluated on it, and
        # the run says so instead of scoring a mismatched mode.
        dev_eval_unavailable = True
    elif dev_source is None:
        for platform, data_spec in data_specs(cfg).items():
            if platform not in spec.platforms:
                continue
            try:
                dev_dataset = open_anchor_dataset(data_spec, "dev")
            except (TrainNotAvailable, FileNotFoundError):
                # A dev partition that cannot be opened leaves the dev curve as
                # not_run for this run; the reason is visible in the manifest.
                dev_eval_unavailable = True
                break
            dev_source = CursorFrameSource(
                dev_dataset,
                micro_batch_frames=spec.micro_batch_frames,
                limit=int(spec.dev_frames),
            )
            datasets[f"dev:{platform}"] = dev_dataset
            break
    dev_eval = None
    if dev_source is not None:
        dev_eval = make_pretrain_dev_eval(
            dev_source,
            device=device,
            micro_batch_frames=spec.micro_batch_frames,
            dev_frames=spec.dev_frames,
        )
    trainer.train(
        until=total_updates,
        log_every=spec.log_every,
        checkpoint_every=min(spec.checkpoint_every, checkpoint_spec.every),
        checkpoint_dir=context.checkpoints,
        checkpoint_kwargs=_checkpoint_kwargs(),
        logger=lambda record: append_jsonl(context.train_log, record),
        dev_eval=dev_eval,
        dev_eval_every=spec.dev_eval_every,
    )
    if checkpoint_spec.save_final:
        trainer.save(context.checkpoints / "final.pt", **_checkpoint_kwargs())
    # Coverage is read after the loop: it counts the frames actually consumed.
    coverage = coverage_report(
        {name: source for name, source in frame_sources.items() if hasattr(source, "coverage")},
        cfg,
        spec,
    )
    return {
        "status": "ok",
        "model": model,
        "trainer": trainer,
        "updates": int(trainer.state.updates),
        "updates_this_resume": int(trainer.state.updates_this_resume),
        "resume_updates": int(trainer.state.resume_updates),
        "checkpoint": str(context.checkpoints / "final.pt"),
        "partition": partition,
        "coverage": coverage,
        "sampler_position": trainer.sampler_state(),
        "dev_eval_unavailable": dev_eval_unavailable,
    }


# ---------------------------------------------------------------------------
# measurement stage
# ---------------------------------------------------------------------------


def build_sensor_model(cfg: n02_config.N02FullConfig, units: str, *, density_verified: bool) -> Any:
    """One T2 ``SensorModel`` for a declared unit family.

    ``density_verified=True`` is only for the synthetic generator, whose joint
    density is declared; the real pseudo-RAW cache cannot verify its noise
    structure, so it is scored as an empirical energy.
    """
    import inspect

    from owac.frameworks.n02_full import sensor

    section = model_section(cfg)
    wanted = {
        "units": str(units),
        "quantisation_step": section.get("quantisation_step"),
        "saturation_level": section.get("saturation_level"),
        "floor_level": section.get("floor_level"),
        "shot_noise_scale": section.get("shot_noise_scale"),
        "read_noise_signal": float(section.get("read_noise_signal", 0.0)),
        "noise_scale": float(section.get("noise_scale", 1.0)),
        "density_verified": bool(density_verified),
    }
    parameters = set(inspect.signature(sensor.SensorModel.__init__).parameters)
    return sensor.SensorModel(**{key: value for key, value in wanted.items() if key in parameters})


def build_measurement_sensor_models(
    cfg: n02_config.N02FullConfig, spec: MeasurementSpec
) -> dict[str, Any]:
    """Every declared unit family of the measurement run, keyed by unit string."""
    models: dict[str, Any] = {}
    for units, verified in (
        (spec.units or _synthetic_units(spec), True),
        (spec.real_units, spec.real_density_verified),
    ):
        if units and units not in models:
            models[str(units)] = build_sensor_model(cfg, str(units), density_verified=verified)
    return models


def build_observation_model(
    cfg: n02_config.N02FullConfig,
    device: str,
    *,
    units: str,
    world_size: int | None = None,
) -> Any:
    """Instantiate the P1 observation stack (T1+T2+T3); else ``not_run``.

    ``units`` is the measurement unit string the ``SensorModel`` requires queries to
    match verbatim; ``density_verified`` stays False for the real pseudo-RAW cache
    (its joint noise structure is not verifiable), so those windows score as
    ``empirical_energy`` rather than being passed off as a proper density.
    """
    needs = (
        ("owac.frameworks.n02_full.observation", "RawObservationEncoder"),
        ("owac.frameworks.n02_full.sensor", "SensorModel"),
        ("owac.frameworks.n02_full.factors", "ObservationModel"),
    )
    missing: list[str] = []
    for module_name, attr in needs:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            missing.append(f"{module_name} ({type(exc).__name__}: {exc})")
            continue
        if not hasattr(module, attr):
            missing.append(f"{module_name}.{attr}")
    if missing:
        raise TrainNotAvailable(
            "the P1 observation stack is incomplete: " + "; ".join(missing), missing=tuple(missing)
        )
    import inspect

    from owac.frameworks.n02_full import factors, observation, sensor

    section = model_section(cfg)
    encoder_wanted = {
        "input_mode": str(section.get("input_mode", "raw3")),
        "in_chans": section.get("in_chans"),
        "depths": tuple(cfg.depths),
        "dims": tuple(cfg.dims),
        "projection_dim": int(section.get("feature_dim", cfg.slot_dim)),
        "alignment": 32,
    }
    encoder_parameters = set(
        inspect.signature(observation.RawObservationEncoder.__init__).parameters
    )
    encoder = observation.RawObservationEncoder(
        **{key: value for key, value in encoder_wanted.items() if key in encoder_parameters}
    )
    sensor_wanted = {
        "units": str(units),
        "quantisation_step": section.get("quantisation_step"),
        "saturation_level": section.get("saturation_level"),
        "floor_level": section.get("floor_level"),
        "shot_noise_scale": section.get("shot_noise_scale"),
        "read_noise_signal": float(section.get("read_noise_signal", 0.0)),
        "noise_scale": float(section.get("noise_scale", 1.0)),
        "density_verified": bool(section.get("density_verified", False)),
    }
    sensor_parameters = set(inspect.signature(sensor.SensorModel.__init__).parameters)
    sensor_model = sensor.SensorModel(
        **{key: value for key, value in sensor_wanted.items() if key in sensor_parameters}
    )
    version = f"p1-{context_hash(cfg)[:12]}"
    model = factors.ObservationModel(
        encoder,
        sensor_model,
        p1_component_config(cfg, world_size=world_size),
        version=version,
    )
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def context_hash(cfg: n02_config.N02FullConfig) -> str:
    return _sha256_text(json.dumps(n02_config.as_dict(cfg), sort_keys=True))


class TaskWindowSource:
    """Controlled P1 tasks (synthetic) or real pseudo-RAW windows for measurement.

    The synthetic branch consumes the frozen generator's per-step
    ``InferenceInput``/``TrainingTargets`` and, when the factors module exposes
    ``build_p1_task``, its assembled task (which carries the explicit
    ``candidate_index`` label). The real branch consumes frozen anchor windows;
    its supervision follows the declared action block only.
    """

    def __init__(
        self,
        *,
        cfg: n02_config.N02FullConfig,
        spec: MeasurementSpec,
        seed: int,
        platform: str | None = None,
        dataset: Any = None,
        kind: str = "synthetic",
    ) -> None:
        self.cfg = cfg
        self.spec = spec
        self.seed = int(seed)
        self.kind = kind
        self.platform = platform
        self.dataset = dataset
        self.units = _synthetic_units(spec) if kind == "synthetic" else str(spec.real_units)
        self.input_mode = _synthetic_input_mode(spec) if kind == "synthetic" else "raw3"
        self.rotations = 0
        self.world_size: int | None = None
        self._pending: Any = None
        self._walker = (
            None
            if dataset is None
            else _AnchorWalker(
                dataset,
                seed=self.seed,
                platform=str(platform),
                partition="train",
                salt="n02-full-p1-measurement-order",
            )
        )
        self._synthetic = None
        self._synthetic_iter = None
        self._task_builder = _load_task_builder()

    # -- state ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "cursor": None}
        if self._walker is not None:
            payload.update(self._walker.state_dict())
        if self._synthetic is not None:
            payload["synthetic_sequences"] = int(self._synthetic.counters.sequences)
        return payload

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if self._walker is not None:
            self._walker.load_state_dict(payload)

    # -- synthetic --------------------------------------------------------------

    def _next_sequence(self, *, skip: set[str] | None = None) -> Any:
        from owac.frameworks.n02_full import synthetic as synth

        if self._synthetic is None:
            sensors = _synthetic_sensors(self.spec.synthetic_sensors)
            self._synthetic = synth.generate_dataset(
                sensors=sensors,
                measurement_model=synth.MeasurementModel(**dict(self.spec.measurement_model)),
                partitions=("train",),
                scenes=int(self.spec.synthetic_scenes),
                frames=int(self.spec.frames_per_episode),
                views_per_scene=1,
                seed=self.seed,
                canvas=tuple(self.spec.canvas),
                action_horizon=int(self.spec.action_horizon),
            )
            self._synthetic_iter = iter(self._synthetic)
        attempts = 0
        while True:
            try:
                sequence = next(self._synthetic_iter)
            except StopIteration:
                self._synthetic_iter = iter(self._synthetic)
                sequence = next(self._synthetic_iter)
            if self.world_size is None:
                return sequence
            # A run pins one object count: the candidate geometry width and the
            # model's candidate capacity must be constant across every batch, and
            # the generator's vocabulary varies the count per scene.
            count = _sequence_world_size(sequence)
            if count != self.world_size:
                continue
            identifier = str(getattr(sequence, "sequence_id", ""))
            if skip is not None and identifier in skip:
                # The scene pool has wrapped around; re-using a scene inside one
                # batch would duplicate acquisition identities (and the ledger would
                # collapse them into one measurement), so it is refused.
                attempts += 1
                if attempts > 2 * max(1, int(self.spec.synthetic_scenes)):
                    raise TrainNotAvailable(
                        "the pinned object count leaves too few synthetic scenes to fill a "
                        f"batch of {self.spec.micro_batch_episodes}: every matching scene is "
                        "already in this batch; raise training.measurement.synthetic_scenes "
                        "or lower micro_batch_episodes"
                    )
                continue
            return sequence

    def probe_world_size(self) -> int | None:
        """Object count of the first synthetic scene, if this source is synthetic."""
        if self.kind != "synthetic":
            return None
        if self.world_size is None:
            sequence = self._next_sequence()
            self._pending = sequence
            self.world_size = _sequence_world_size(sequence)
        return self.world_size

    def _synthetic_batch(self) -> MeasurementBatch:
        from owac.frameworks.n02_full.state import SequenceState

        episodes = []
        used: set[str] = set()
        for _ in range(self.spec.micro_batch_episodes):
            if self._pending is not None:
                sequence = self._pending
                self._pending = None
            else:
                sequence = self._next_sequence(skip=used)
            used.add(str(getattr(sequence, "sequence_id", "")))
            episodes.append(sequence)
        # The correct candidate's position rotates between micro-batches: the task
        # must not be solvable by a fixed position, while every row of one batch
        # still shares one explicit label (StepOutput exposes one world score per
        # batch, not one per row).
        correct_index = self.rotations % 4
        self.rotations += 1
        steps = []
        for frame_index in range(self.spec.frames_per_episode):
            bundles = []
            targets = []
            dictionary = None
            for episode in episodes:
                if self._task_builder is not None:
                    task = self._task_builder(
                        episode, frame_index=frame_index, correct_index=correct_index
                    )
                    bundles.append(task.inference.bundle)
                    targets.append(task.targets)
                    dictionary = task.dictionary if dictionary is None else dictionary
                else:
                    bundles.append(episode.inference[frame_index].bundle)
                    targets.append(episode.targets[frame_index])
            steps.append(
                MeasurementStep(
                    bundle=merge_bundles(bundles),
                    targets=_merge_targets(targets),
                    state=SequenceState(),
                    dictionary=dictionary,
                )
            )
        return MeasurementBatch(
            steps=steps,
            kind="synthetic",
            units=self.units,
            candidate_position=correct_index,
        )

    # -- real -------------------------------------------------------------------

    def _real_batch(self) -> MeasurementBatch:
        from owac.frameworks.n02_full import data as n02_data
        from owac.frameworks.n02_full.state import SequenceState

        assert self._walker is not None
        samples = [
            self.dataset[self._walker.next_index()] for _ in range(self.spec.micro_batch_episodes)
        ]
        bundles = []
        targets = []
        for sample in samples:
            inference, sample_targets = n02_data.sample_to_inference_and_targets(sample)
            bundles.append(inference.bundle)
            targets.append(sample_targets)
        return MeasurementBatch(
            steps=[
                MeasurementStep(
                    bundle=merge_bundles(bundles),
                    targets=_merge_targets(targets),
                    state=SequenceState(),
                )
            ],
            kind="real",
            units=self.units,
        )

    def next_micro_batch(self) -> MeasurementBatch:
        if self.kind == "real":
            return self._real_batch()
        return self._synthetic_batch()


def _merge_targets(targets: Sequence[Any]) -> Any:
    """Batch per-episode TrainingTargets; only supervision enters, never truth."""
    from owac.frameworks.n02_full.state import TrainingTargets

    np = _numpy()

    def _cat(name: str) -> Any:
        values = [getattr(target, name) for target in targets]
        if any(value is None for value in values):
            return None
        try:
            return np.concatenate([np.asarray(value)[None, ...] for value in values], axis=0)
        except ValueError:
            return None

    geometry: Any = None
    if all(isinstance(getattr(target, "optional_geometry", None), Mapping) for target in targets):
        geometry = {}
        for key in targets[0].optional_geometry:
            values = [target.optional_geometry[key] for target in targets]
            try:
                stacked = np.stack([np.asarray(value) for value in values], axis=0)
            except ValueError:
                geometry = None
                break
            if all(np.asarray(value).shape == () for value in values):
                # A scalar label/flag collapses to one value for the batch: the
                # controlled task fixes one correct candidate per batch, and a
                # batch that mixed labels is refused rather than silently scored
                # with a single row's label.
                unique = {value.item() for value in stacked.reshape(-1)}
                if key == "candidate_index" and len(unique) != 1:
                    geometry = None
                    break
                geometry[key] = stacked[0].item()
            else:
                geometry[key] = stacked
    return TrainingTargets(
        actions=_cat("actions"),
        action_mask=_cat("action_mask"),
        optional_geometry=geometry,
        optional_relations=None,
    )


def _sequence_world_size(sequence: Any) -> int:
    """Number of objects of one synthetic sequence, from its declared truth geometry."""
    np = _numpy()
    world_px = np.asarray(sequence.truth[0].geometry["world_px"])
    return int(world_px.reshape(-1, 2).shape[0])


def _synthetic_input_mode(spec: MeasurementSpec) -> str:
    """The T1 input mode the declared synthetic measurement model needs."""
    return "cfa4" if str(spec.measurement_model.get("kind", "cfa")) == "cfa" else "raw3"


def _synthetic_units(spec: MeasurementSpec) -> str:
    """The declared unit string of the synthetic measurement model (T2 contract)."""
    from owac.frameworks.n02_full import synthetic as synth

    return str(synth.MeasurementModel(**dict(spec.measurement_model)).unit())


def _synthetic_sensors(entry: Any) -> tuple[Any, ...]:
    from owac.frameworks.n02_full import synthetic as synth

    if not entry:
        return (
            synth.SensorSpec(
                camera_id="cam0",
                height=64,
                width=64,
                input_mode="bayer_cfa",
                cfa_pattern="RGGB",
                black_level=64.0,
                white_level=4095.0,
                bit_depth=12,
                read_noise_e=2.0,
                gain=1.0,
                exposure_ms=1.0,
            ),
        )
    sensors = []
    for item in entry:
        if not isinstance(item, Mapping):
            raise TrainingSpecError("training.measurement.synthetic_sensors: expected objects")
        sensors.append(synth.SensorSpec(**dict(item)))
    return tuple(sensors)


def _load_task_builder() -> Any:
    try:
        module = importlib.import_module("owac.frameworks.n02_full.factors")
    except ImportError:
        return None
    return getattr(module, "build_p1_task", None)


@dataclass(slots=True)
class MeasurementStep:
    bundle: Any
    targets: Any
    state: Any
    history: Any = None
    ledger: Any = None
    budget: Any = None
    dictionary: Any = None


@dataclass(slots=True)
class MeasurementBatch:
    steps: list
    kind: str = "synthetic"
    units: str = ""
    candidate_position: int | None = None


class MeasurementSource:
    """Alternate the declared measurement sources, one micro-batch at a time.

    One encoder has exactly one input mode (T1 never converts raw3 <-> cfa4), so a
    declared source whose frames need another mode cannot share this run; it is
    recorded in ``skipped_sources`` with its reason instead of being mixed in.
    """

    def __init__(
        self,
        sources: Sequence[TaskWindowSource],
        *,
        skipped: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if not sources:
            raise TrainNotAvailable("measurement sources are empty")
        self.sources = list(sources)
        self.skipped_sources = [dict(entry) for entry in skipped]
        self._cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": int(self._cursor),
            "sources": [source.state_dict() for source in self.sources],
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        self._cursor = int(payload.get("cursor", 0))
        for source, state in zip(self.sources, payload.get("sources", []), strict=False):
            source.load_state_dict(state)

    def next_micro_batch(self) -> MeasurementBatch:
        source = self.sources[self._cursor % len(self.sources)]
        self._cursor += 1
        return source.next_micro_batch()

    def probe_world_size(self) -> int | None:
        for source in self.sources:
            probed = source.probe_world_size()
            if probed is not None:
                return probed
        return None


def build_measurement_sources(
    cfg: n02_config.N02FullConfig, spec: MeasurementSpec, seed: int
) -> MeasurementSource:
    """Open the declared measurement sources (controlled tasks and/or real RAW).

    One encoder holds exactly one T1 input mode, so a source whose frames need
    another mode is *not* constructed: it is recorded in ``skipped_sources`` with
    its reason. The run manifest's ``mode_gate`` block is derived from this same
    decision, so the receipt and the run can never disagree.
    """
    encoder_mode = str(model_section(cfg).get("input_mode", "raw3"))
    sources: list[TaskWindowSource] = []
    skipped: list[dict[str, Any]] = []
    for kind in spec.sources:
        if kind == "synthetic":
            source = TaskWindowSource(cfg=cfg, spec=spec, seed=seed, kind="synthetic")
            if source.input_mode != encoder_mode:
                skipped.append(
                    {
                        "kind": kind,
                        "declared_mode": source.input_mode,
                        "encoder_mode": encoder_mode,
                        "reason": (
                            "the synthetic frames need another T1 input mode and the frozen "
                            "contract never converts raw3 <-> cfa4"
                        ),
                    }
                )
                continue
            sources.append(source)
            continue
        # The mode is decided before the dataset is opened: a gated-out source must not
        # even require the cache to be readable.
        real_mode = "raw3"
        specs = data_specs(cfg)
        if real_mode != encoder_mode:
            skipped.append(
                {
                    "kind": kind,
                    "platform": sorted(specs)[0] if specs else "",
                    "declared_mode": real_mode,
                    "encoder_mode": encoder_mode,
                    "reason": (
                        "the real 3-channel pseudo-RAW needs the raw3 mode; run it as its own "
                        "measurement run instead of mixing modes in one encoder"
                    ),
                }
            )
            continue
        if not specs:
            raise TrainNotAvailable("training.data declares no platform for the real-window source")
        platform = sorted(specs)[0]
        dataset = open_anchor_dataset(specs[platform], "train")
        sources.append(
            TaskWindowSource(
                cfg=cfg, spec=spec, seed=seed, platform=platform, dataset=dataset, kind="real"
            )
        )
    if not sources:
        raise TrainNotAvailable(
            "every declared measurement source needs a different input mode than the encoder: "
            + "; ".join(str(entry["reason"]) for entry in skipped)
        )
    return MeasurementSource(sources, skipped=skipped)


def measurement_loss_fn(
    model: Any, spec: MeasurementSpec, *, sensor_models: Mapping[str, Any] | None = None
) -> Any:
    """Build the loss callable that drives ``step_bundle`` over one task batch."""
    try:
        module = importlib.import_module("owac.frameworks.n02_full.model")
    except ImportError as exc:
        raise TrainNotAvailable(
            f"owac.frameworks.n02_full.model is not available: {type(exc).__name__}: {exc}",
            missing=("owac.frameworks.n02_full.model",),
        ) from exc
    step_bundle = getattr(module, "step_bundle", None)
    if step_bundle is None:
        raise TrainNotAvailable("owac.frameworks.n02_full.model.step_bundle is missing")

    weights = {
        "raw_proper_prediction_prequential": float(spec.loss_weights["raw_proper_prediction"]),
        "raw_proper_prediction_fit": 0.0,
        "candidate_ce": float(spec.loss_weights["candidate_ce"]),
        "geometry_aux": float(spec.loss_weights["geometry_aux"]),
    }
    history_builder = _load_history_builder()

    def _loss(model_ref: Any, batch: MeasurementBatch, info: MicroInfo) -> tuple[Any, dict, dict]:
        torch = _torch()
        replacement = (sensor_models or {}).get(batch.units)
        previous_sensor = getattr(model_ref, "sensor_model", None)
        swapped = replacement is not None and replacement is not previous_sensor
        if swapped:
            model_ref.sensor_model = replacement
        total = None
        merged: dict[str, dict[str, Any]] = {}
        previous = None
        if getattr(model_ref, "candidates", None) is None:
            # The online path reads the candidate worlds from the bound public
            # dictionary; the controlled task constructor owns that dictionary, so
            # the harness binds the one it assembled instead of inventing a second.
            for candidate_step in batch.steps:
                if getattr(candidate_step, "dictionary", None) is not None:
                    device = None
                    parameters = model_parameters(model_ref)
                    if parameters:
                        device = parameters[0].device
                    model_ref.bind_dictionary(move_dictionary_to(candidate_step.dictionary, device))
                    break
        # One batched history encode for the whole micro-batch instead of one per
        # step: the history frames of every step are encoded in a single forward and
        # sliced back per step. ConvNeXt V2 normalises per sample, so this is the same
        # arithmetic; it is skipped (per-step path) whenever the row layout is not
        # aligned or the pyramid cannot be sliced.
        precomputed = None
        if history_builder is not None and not _batched_history_disabled():
            precomputed = _batched_history_contexts(history_builder, model_ref, batch.steps)
        for index, step in enumerate(batch.steps):
            history = None
            if history_builder is not None:
                if precomputed is not None:
                    history = precomputed[index]
                else:
                    history = history_builder(model_ref, previous, step, index)
            output = step_bundle(
                step.bundle,
                step.state,
                model_ref,
                history,
                ledger=step.ledger,
                budget=step.budget,
                targets=step.targets,
            )
            value, terms = loss_terms_from_step(
                output,
                step.targets,
                weights=weights,
                ce_temperature=spec.ce_temperature,
            )
            if value is not None:
                total = value if total is None else total + value
            for name, record in terms.items():
                entry = merged.setdefault(
                    name,
                    {
                        "status": record.get("status", "unavailable"),
                        "value": 0.0,
                        "sum": 0.0,
                        "activation_count": 0,
                        "micro_batches": 0,
                        "reason": record.get("reason", ""),
                        "kind": record.get("kind", ""),
                    },
                )
                tensor = record.get("tensor")
                if tensor is not None and torch.is_tensor(tensor):
                    existing = entry.get("tensor")
                    entry["tensor"] = tensor if existing is None else existing + tensor
                if record.get("status") in ("ok", "detached"):
                    entry["value"] += float(record.get("value") or 0.0)
                    entry["sum"] += float(record.get("sum") or 0.0)
                    entry["activation_count"] += int(record.get("activation_count") or 0)
                    entry["micro_batches"] += 1
                elif entry["status"] in ("ok", "detached"):
                    entry["status"] = record["status"]
                    entry["reason"] = record.get("reason", "")
                for key in ("label_source", "score_kind", "gradient_available"):
                    if key in record:
                        entry[key] = record[key]
            previous = step.bundle
        if swapped:
            model_ref.sensor_model = previous_sensor
        return (
            total,
            merged,
            {
                "steps": len(batch.steps),
                "kind": batch.kind,
                "units": batch.units,
                "candidate_position": batch.candidate_position,
            },
        )

    return _loss


def move_dictionary_to(dictionary: Any, device: Any) -> Any:
    """Move a declared candidate dictionary's geometry tensors to ``device``.

    ``build_p1_task`` assembles the dictionary on CPU from generator truth; on a
    CUDA run the candidate encoder would otherwise receive a CPU vector. Moving the
    geometry keeps the declared hypotheses identical (same values, same order) and
    is recorded in the term as a device adaptation, not a new candidate set.
    """
    torch = _torch()
    if not str(device).startswith("cuda"):
        return dictionary
    for spec in tuple(getattr(dictionary, "specs", ()) or ()):
        geometry = getattr(spec, "geometry", None)
        if geometry is None:
            continue
        for name in ("world_px", "scale_px", "depth"):
            value = getattr(geometry, name, None)
            if torch.is_tensor(value) and value.device != torch.device(device):
                setattr(geometry, name, value.to(device))
    return dictionary


def _batched_history_disabled() -> bool:
    """The batched history encode is OFF by default.

    An A/B run (3 updates, seed 1103, bf16) showed it is NOT equivalent to the per-step
    path: max |Δloss| 11.78, max |Δgrad_base| 6.2e-3 against per-step 82.94/83.91/86.71
    versus batched 83.42/81.97/98.49. It is therefore opt-in only, so no formal run can
    silently pick it up. `OWAC_BATCHED_HISTORY=1` enables it for debugging.
    """
    return os.environ.get("OWAC_BATCHED_HISTORY", "") in ("", "0")


def _slice_level(level: Any, start: int, stop: int) -> Any:
    """Rows ``[start:stop)`` of one feature level, every row-aligned field included.

    ``FeatureLevel`` carries ``values`` *plus* ``valid`` and ``sensor_coordinates``; slicing
    only ``values`` leaves a mask of the wrong length, which is what made the first A/B run
    of this optimisation disagree with the per-step path.
    """
    import dataclasses as _dc

    updates: dict[str, Any] = {}
    for name in ("values", "valid", "sensor_coordinates"):
        item = getattr(level, name, None)
        if item is None:
            continue
        if not hasattr(item, "__getitem__"):
            raise TypeError(f"feature level field {name!r} is not sliceable")
        length = getattr(item, "shape", None)
        if length is None or len(length) == 0 or int(length[0]) > stop:
            updates[name] = item[start:stop]
    if "values" not in updates:
        raise TypeError("feature level has no values")
    try:
        return _dc.replace(level, **updates)
    except (TypeError, ValueError):
        clone = copy.copy(level)
        for name, item in updates.items():
            setattr(clone, name, item)
        return clone


def _batched_history_contexts(builder: Any, model: Any, steps: Sequence[Any]) -> list[Any] | None:
    """History contexts for every step, from one encode over all history frames.

    The steps must share one row layout for the per-step slicing to be valid; when they
    do not (or the pyramid cannot be sliced), this returns ``None`` and the caller falls
    back to the per-step path. ConvNeXt V2 normalises per sample, so the batched encode
    computes the same arithmetic as the per-step one.
    """
    if len(steps) < 2:
        return None
    layouts = [tuple(step.bundle.episode_ids) for step in steps]
    if any(layout != layouts[0] for layout in layouts[1:]):
        return None
    rows = len(layouts[0])
    if rows == 0:
        return None
    try:
        from owac.frameworks.n02_full import factors

        context_cls = factors.HistoryContext
    except ImportError:
        return None
    try:
        pyramid = model.encode(merge_bundles([step.bundle for step in steps[1:]]))
    except Exception:  # noqa: BLE001 - the batched path is an optimisation only
        return None
    try:
        pooled_all = pyramid.levels["stride16"].values
    except (AttributeError, KeyError):
        return None
    contexts: list[Any] = []
    for index, step in enumerate(steps):
        if index == 0:
            contexts.append(builder(model, None, step, index))
            continue
        start, stop = (index - 1) * rows, index * rows
        try:
            levels = {
                name: _slice_level(level, start, stop) for name, level in pyramid.levels.items()
            }
        except TypeError:
            return None
        pooled = pooled_all[start:stop]
        pooled = pooled.mean(dim=(-1, -2)) if hasattr(pooled, "mean") else pooled
        contexts.append(
            context_cls(
                pooled_features=pooled,
                levels=levels,
                episode_ids=tuple(step.bundle.episode_ids),
                anchor_ids=tuple(step.bundle.anchor_ids),
                last_frame_index=index - 1,
                eta=None,
            )
        )
    return contexts


def _load_history_builder() -> Any:
    """Pool the previous step's features into a causal ``HistoryContext``."""
    try:
        factors = importlib.import_module("owac.frameworks.n02_full.factors")
    except ImportError:
        return None
    context_cls = getattr(factors, "HistoryContext", None)
    if context_cls is None:
        return None
    torch = _torch()

    def _build(model: Any, previous_bundle: Any, step: Any, index: int) -> Any:
        batch = len(step.bundle.episode_ids) or None
        if previous_bundle is None:
            width = 512
            parameters = model_parameters(model)
            device = parameters[0].device if parameters else torch.device("cpu")
            pooled = torch.zeros((batch or 1, width), dtype=torch.float32, device=device)
            return context_cls(
                pooled_features=pooled,
                levels=None,
                episode_ids=tuple(step.bundle.episode_ids),
                anchor_ids=tuple(step.bundle.anchor_ids),
                last_frame_index=None,
                eta=None,
            )
        pyramid = model.encode(previous_bundle)
        level = pyramid.levels.get("stride16") or next(iter(pyramid.levels.values()))
        values = level.values
        pooled = values.mean(dim=(-1, -2)) if hasattr(values, "mean") else values
        return context_cls(
            pooled_features=pooled,
            levels=pyramid.levels,
            episode_ids=tuple(step.bundle.episode_ids),
            anchor_ids=tuple(step.bundle.anchor_ids),
            last_frame_index=int(index) - 1,
            eta=None,
        )

    return _build


class CandidateRatioEstimator(_torch().nn.Module):
    """``log r_phi(theta, x)`` over the shared, frozen Base features.

    ``theta`` is a candidate's declared geometry/nuisance vector and ``x`` the pooled
    Base features of the observation. The forward accepts the ``(2, N, D)`` stacked
    shape the vendored official losses use and returns both log-ratios, so
    ``owac.baselines.nre_losses.NRELoss``/``BNRELoss`` drive it unchanged.
    """

    def __init__(self, feature_dim: int, candidate_dim: int, hidden: int = 128) -> None:
        super().__init__()
        nn = _torch().nn
        self.theta_net = nn.Sequential(
            nn.Linear(int(candidate_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.x_net = nn.Sequential(
            nn.Linear(int(feature_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)

    def _score(self, theta: Any, shared: Any) -> Any:
        return self.head(self.theta_net(theta) + shared).squeeze(-1)

    def forward(self, theta: Any, x: Any) -> Any:
        shared = self.x_net(x)
        if theta.dim() == 3:
            return self._score(theta[0], shared), self._score(theta[1], shared)
        return self._score(theta, shared)


def ratio_arm_pairs(
    cfg: n02_config.N02FullConfig,
    spec: MeasurementSpec,
    model: Any,
    *,
    seed: int,
    batch: int,
) -> tuple[Any, Any, list[Any]]:
    """One batch of ``(theta_correct, x)`` pairs plus the tasks they came from.

    The Base is frozen and shared: ``x`` is its pooled pooled feature of the task's
    bundle, and ``theta`` is the *correct* candidate's declared vector. Negatives are
    the within-batch roll the official losses perform, so no extra sampling rule is
    invented here.
    """
    from owac.frameworks.n02_full import factors
    from owac.frameworks.n02_full import synthetic as synth

    torch = _torch()
    measurement = dict((cfg.training or {}).get("measurement", {}) or {})
    sensors = tuple(
        synth.SensorSpec(**dict(entry)) for entry in measurement.get("synthetic_sensors", [])
    )
    if not sensors:
        raise TrainNotAvailable("training.measurement.synthetic_sensors is empty")
    dataset = synth.generate_dataset(
        sensors=sensors,
        measurement_model=synth.MeasurementModel(
            **dict(measurement.get("measurement_model", {"kind": "cfa"}))
        ),
        partitions=("train",),
        scenes=max(64, int(batch) * 4),
        frames=max(3, int(measurement.get("frames_per_episode", 8))),
        views_per_scene=1,
        seed=int(seed),
        canvas=tuple(measurement.get("canvas", (64, 64))),
        action_horizon=int(measurement.get("action_horizon", 4)),
    )
    thetas: list[Any] = []
    features: list[Any] = []
    tasks: list[Any] = []
    for sequence in dataset:
        task = factors.build_p1_task(sequence, frame_index=1)
        index = task.candidate_index
        if index is None:
            continue
        with torch.no_grad():
            pyramid = model.encode(task.inference.bundle)
            pooled = factors.pool_feature_pyramid(pyramid)
        geometry = task.dictionary.specs[index].geometry
        thetas.append(geometry.vector().detach().to(torch.float32))
        features.append(pooled.detach().to(torch.float32).reshape(-1))
        tasks.append(task)
        if len(thetas) >= int(batch):
            break
    if len(thetas) < 2:
        raise TrainNotAvailable("the ratio arm needs at least two controlled tasks per batch")
    width = min(int(item.numel()) for item in thetas)
    return (
        torch.stack([item[:width] for item in thetas]),
        torch.stack([item[: min(item.numel() for item in features)] for item in features]),
        tasks,
    )


def run_ratio_arm(
    cfg: n02_config.N02FullConfig,
    spec: MeasurementSpec,
    context: RunContext,
    *,
    arm: str,
    updates: int,
    device: str,
    base_checkpoint: str = "",
    batch: int = 16,
    log_every: int = 50,
    checkpoint_spec: CheckpointSpec | None = None,
) -> dict[str, Any]:
    """Bounded first round of the official NRE/BNRE arms over the frozen shared Base."""
    torch = _torch()
    bound_torch_threads()
    losses_module = importlib.import_module("owac.baselines.nre_losses")
    official = getattr(losses_module, "NRELoss" if arm == "nre" else "BNRELoss", None)
    if official is None:
        raise TrainNotAvailable(f"owac.baselines.nre_losses has no {arm.upper()} loss")
    pinned = "cbf500c09c538a014e72477d166acc62a6d8409a"
    model = build_observation_model(
        cfg, device, units=spec.units or _synthetic_units(spec), world_size=None
    )
    base_inheritance: dict[str, Any] = {}
    if base_checkpoint:
        payload = load_checkpoint(base_checkpoint)
        # A pretraining checkpoint carries FCMAE-only modules (mask_token, the FCMAE
        # decoder) that the measurement model does not have; the Base is inherited
        # tensor-by-tensor with an explicit matched/skipped report instead of a strict
        # load that would reject the whole checkpoint.
        base_inheritance = load_base_subset(model, payload["model"])
        base_inheritance["source_stage"] = str(payload.get("stage", ""))
        base_inheritance["source_updates"] = int(payload.get("updates", 0) or 0)
        base_inheritance["source_checkpoint"] = str(base_checkpoint)
    for parameter in model_parameters(model):
        parameter.requires_grad_(False)
    # The declared candidate vector width is read from one real batch instead of being
    # derived from a config guess: geometry vectors are (2 + 1 + 1) * objects wide.
    probe_theta, probe_features, _probe_tasks = ratio_arm_pairs(
        cfg, spec, model, seed=context.seed, batch=batch
    )
    feature_dim = int(probe_features.shape[1])
    candidate_dim = int(probe_theta.shape[1])
    estimator = CandidateRatioEstimator(feature_dim, candidate_dim).to(device)
    loss_fn = official(estimator) if arm == "nre" else official(estimator, 100.0)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    for update in range(1, int(updates) + 1):
        if update == 1:
            theta, features = probe_theta, probe_features
        else:
            theta, features, _tasks = ratio_arm_pairs(
                cfg, spec, model, seed=context.seed + update, batch=batch
            )
        theta = theta.to(device)[:, :candidate_dim]
        features = features.to(device)
        loss = loss_fn(theta, features)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        stats = gradient_statistics(estimator)
        optimizer.step()
        if update % max(1, log_every) == 0 or update == int(updates):
            record = {
                "update": update,
                "loss": float(loss.detach().item()),
                "lr": [float(group["lr"]) for group in optimizer.param_groups],
                "terms": {
                    arm: {
                        "status": "ok",
                        "value": float(loss.detach().item()),
                        "sum": float(loss.detach().item()),
                        "activation_count": int(theta.shape[0]),
                        "reason": "",
                        "kind": f"official_{arm}_loss",
                    }
                },
                "time": _now(),
                **stats,
            }
            history.append(_json_ready(record))
            append_jsonl(context.train_log, record)
    checkpoint_spec = checkpoint_spec or CheckpointSpec()
    if checkpoint_spec.save_final:
        context.checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "stage": context.stage,
                "arm": arm,
                "seed": context.seed,
                "updates": int(updates),
                "estimator": estimator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng_state(),
                "base_checkpoint": base_checkpoint,
                "base_frozen": True,
                "partition": {"kind": "ratio-arm", "arm": arm, "batch": int(batch)},
                "config_hash": context.config_hash,
                "versions": interpreter_info(),
            },
            context.checkpoints / "final.pt",
        )
    return {
        "status": "ok",
        "trainer": type(
            "_ArmState",
            (),
            {
                "state": type(
                    "_S",
                    (),
                    {
                        "history": history,
                        "updates": updates,
                        "updates_this_resume": updates,
                        "resume_updates": 0,
                    },
                )()
            },
        )(),
        "updates": int(updates),
        "updates_this_resume": int(updates),
        "resume_updates": 0,
        "checkpoint": str(context.checkpoints / "final.pt"),
        "partition": {"kind": "ratio-arm", "arm": arm},
        "arm_manifest": {
            "ablation_arm": arm,
            "loss_source": f"owac.baselines.nre_losses:{'NRELoss' if arm == 'nre' else 'BNRELoss'}",
            "pinned_commit": pinned,
            "hyperparameters": {} if arm == "nre" else {"lambda": 100.0},
            "base_frozen": True,
            "base_checkpoint": base_checkpoint,
            "base_inheritance": base_inheritance,
            "same_base": True,
            "same_history_and_labels": True,
            "same_budget_as_main": False,
            "budget_deviation": (
                f"bounded first round ({updates} updates, batch {batch}); not the §5 "
                "same-training-budget comparison"
            ),
        },
    }


def arm_report(spec: MeasurementSpec, arm: str, *, updates: int | None = None) -> dict[str, Any]:
    """The declared budget of one ablation arm and its deviation from P1 §5."""
    weights = dict(getattr(spec, "loss_weights", {}) or {})
    if arm in ARM_LOSS_WEIGHTS:
        weights = dict(ARM_LOSS_WEIGHTS[arm])
    declared = {"raw_proper_prediction": 1.0, "candidate_ce": 1.0, "geometry_aux": 0.1}
    return {
        "arm": arm,
        "loss_weights": weights,
        "frozen_weights": declared,
        "same_budget_as_main": False,
        "budget_deviation": (
            "bounded first round ("
            + str(updates if updates is not None else spec.updates)
            + " updates), not the §5 same-training-budget comparison; the first round is a "
            "feasibility signal only"
        ),
        "missing_interface": ARM_MISSING_INTERFACES.get(arm, ""),
        "status": "not_run" if arm in ARM_MISSING_INTERFACES else "runnable",
    }


def run_measurement(
    cfg: n02_config.N02FullConfig,
    spec: MeasurementSpec,
    context: RunContext,
    *,
    lr_scale: float = 1.0,
    updates: int | None = None,
    device: str = "cpu",
    resume: Mapping[str, Any] | None = None,
    model_factory: Any = None,
    sources: Any = None,
    checkpoint_spec: CheckpointSpec | None = None,
) -> dict[str, Any]:
    """Driving loop of ``--stage p1-measurement``."""
    torch = _torch()
    additional = updates is not None
    start = budget_start(resume, "p1-measurement")
    total_updates = (start + int(updates)) if additional else int(spec.updates)
    if total_updates <= start:
        raise TrainNotAvailable(
            f"nothing to train: resume at {start} updates and budget {total_updates}"
        )
    units = spec.units or _synthetic_units(spec)
    if sources is None:
        sources = build_measurement_sources(cfg, spec, context.seed)
    probed_world_size = sources.probe_world_size() if hasattr(sources, "probe_world_size") else None
    if model_factory is None:

        def model_factory() -> Any:
            return build_observation_model(cfg, device, units=units, world_size=probed_world_size)

    model = model_factory()
    model = move_model_to(model, device)
    if not model_parameters(model):
        raise TrainNotAvailable("the observation model exposes no trainable parameters")
    base_names, head_names = split_parameter_names(
        [name for name, _ in model_named_parameters(model)]
    )
    by_name = dict(model_named_parameters(model))
    groups = [
        {"params": [by_name[name] for name in base_names], "lr": spec.lr_base * float(lr_scale)},
        {"params": [by_name[name] for name in head_names], "lr": spec.lr_head * float(lr_scale)},
    ]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, betas=tuple(spec.betas), weight_decay=spec.weight_decay)
    trainer = P1Trainer(
        stage=context.stage,
        seed=context.seed,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        source=sources,
        loss_fn=measurement_loss_fn(
            model, spec, sensor_models=build_measurement_sensor_models(cfg, spec)
        ),
        updates=total_updates,
        accum_steps=spec.accum_steps,
        base_lrs=[float(group["lr"]) for group in optimizer.param_groups],
        warmup_updates=spec.warmup_updates(total_updates),
        min_lr_ratio=spec.min_lr_ratio,
        clip_grad=spec.grad_clip,
        device=device,
        amp=spec.amp,
        grad_conflict_every=spec.grad_conflict_every,
    )
    if resume is not None:
        trainer.resume_from(resume)
    checkpoint_spec = checkpoint_spec or CheckpointSpec()
    partition = {
        "kind": "p1-measurement",
        "sources": list(spec.sources),
        "config_hash": context.config_hash,
        "seed": context.seed,
    }

    def _checkpoint_kwargs() -> dict[str, Any]:
        return {
            "partition": partition,
            "config_hash": context.config_hash,
            "run_id": context.run_id,
            "extras": {"lr_scale": float(lr_scale)},
        }

    trainer.train(
        until=total_updates,
        log_every=spec.log_every,
        checkpoint_every=min(spec.checkpoint_every, checkpoint_spec.every),
        checkpoint_dir=context.checkpoints,
        checkpoint_kwargs=_checkpoint_kwargs(),
        logger=lambda record: append_jsonl(context.train_log, record),
    )
    final_receipt: dict[str, Any] = {"status": "skipped"}
    if checkpoint_spec.save_final:
        final_receipt = save_final_checkpoint(
            trainer, context.checkpoints / "final.pt", **_checkpoint_kwargs()
        )
    return {
        "status": "ok",
        "model": model,
        "trainer": trainer,
        "updates": int(trainer.state.updates),
        "updates_this_resume": int(trainer.state.updates_this_resume),
        "resume_updates": int(trainer.state.resume_updates),
        "checkpoint": str(context.checkpoints / "final.pt"),
        "final_checkpoint": final_receipt,
        "partition": partition,
        "skipped_sources": list(getattr(sources, "skipped_sources", [])),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m owac.training.n02_full",
        description=(
            "P1 training CLI for the N02 full line. A missing module, asset or checkpoint is "
            "reported as status=not_run with a reason and exit code 0; nothing unverified is "
            "reported as a training result."
        ),
    )
    parser.add_argument("--stage", choices=STAGES, required=True, help="which P1 stage to run")
    parser.add_argument("--config", default=None, help="config JSON path")
    parser.add_argument(
        "--seed", type=int, default=None, help="model seed (default: the config's first seed)"
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help=(
            "update budget; without --resume it is the total budget, with --resume it is the "
            "number of additional updates (cumulative = resume + new)"
        ),
    )
    parser.add_argument("--lr-scale", type=float, default=1.0, help="dev-grid learning-rate scale")
    parser.add_argument(
        "--output", default=None, help="output root (default: config paths.output_root)"
    )
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument(
        "--device", default=None, help="torch device (default: config runtime.device)"
    )
    parser.add_argument(
        "--tag", default="", help="run-id suffix (e.g. lr0.5) keeping dev-grid runs distinct"
    )
    parser.add_argument("--log-every", type=int, default=None, help="override the log interval")
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="override micro_batch_frames / micro_batch_episodes (effective batch is unchanged)",
    )
    parser.add_argument(
        "--amp",
        default=None,
        choices=("off", "tf32", "bf16"),
        help="override the declared numerical mode for this run",
    )
    parser.add_argument(
        "--arm",
        default="main",
        choices=MEASUREMENT_ARMS,
        help="measurement ablation arm (bounded first round; see arm_report)",
    )
    parser.add_argument(
        "--synthetic-frames",
        action="store_true",
        help="pretrain the stem on the controlled generator's CFA4 frames instead of the cache",
    )
    parser.add_argument(
        "--precision-check",
        action="store_true",
        help=(
            "run the declared reduced-precision error check against the FP32 reference "
            "instead of training; the receipt goes to the run directory"
        ),
    )
    parser.add_argument(
        "--precision-micro-batch",
        type=int,
        default=None,
        help="micro-batch size of the precision check (default: the stage micro-batch)",
    )
    parser.add_argument(
        "--platforms",
        default=None,
        help="comma-separated platform override for the pretraining frame source",
    )
    parser.add_argument(
        "--dev-eval-every",
        type=int,
        default=None,
        help="override the fixed dev-evaluation interval (updates)",
    )
    return parser


def _resolve_specs(cfg: n02_config.N02FullConfig, stage: str) -> tuple[Any, CheckpointSpec]:
    training = cfg.training if isinstance(cfg.training, dict) else {}
    checkpoint = _section(CheckpointSpec, training.get("checkpoint", {}), "training.checkpoint")
    if stage == "p1-pretrain":
        return _section(PretrainSpec, training.get("pretrain", {}), "training.pretrain"), checkpoint
    return (
        _section(MeasurementSpec, training.get("measurement", {}), "training.measurement"),
        checkpoint,
    )


def mode_gate_payload(spec: Any, cfg: n02_config.N02FullConfig) -> dict[str, Any]:
    """Declared input mode and the per-source gate outcome (never a silent skip)."""
    encoder_mode = str(model_section(cfg).get("input_mode", "raw3"))
    payload: dict[str, Any] = {"encoder_mode": encoder_mode, "sources": []}
    if isinstance(spec, MeasurementSpec):
        for kind in spec.sources:
            if kind == "synthetic":
                required = _synthetic_input_mode(spec)
                payload["sources"].append(
                    {
                        "kind": "synthetic",
                        "required_mode": required,
                        "units": spec.units or _synthetic_units(spec),
                        "gate": "active" if required == encoder_mode else "skipped",
                    }
                )
            else:
                payload["sources"].append(
                    {
                        "kind": "real",
                        "required_mode": "raw3",
                        "units": spec.real_units,
                        "density": (
                            "proper_nll" if spec.real_density_verified else "empirical_energy"
                        ),
                        "gate": "active" if encoder_mode == "raw3" else "skipped",
                    }
                )
    else:
        payload["sources"].append(
            {
                "kind": "real-pretrain",
                "required_mode": encoder_mode,
                "gate": "active",
                "platforms": list(getattr(spec, "platforms", ())),
            }
        )
    return payload


def _base_payload(
    args: argparse.Namespace, cfg: n02_config.N02FullConfig, context: RunContext
) -> dict[str, Any]:
    payload = context.as_dict()
    payload.update(
        {
            "status": "not_run",
            "reason": "",
            "missing": [],
            "interpreter": interpreter_info(),
            "git_sha": _git_sha(),
            "started_at": _now(),
            "device": _device_for(args, cfg),
            "lr_scale": float(args.lr_scale),
            "input_mode": str(model_section(cfg).get("input_mode", "raw3")),
            "config": n02_config.as_dict(cfg),
        }
    )
    return payload


def _device_for(args: argparse.Namespace, cfg: n02_config.N02FullConfig) -> str:
    if args.device:
        return str(args.device)
    runtime = cfg.runtime if isinstance(cfg.runtime, dict) else {}
    return str(runtime.get("device", "cpu"))


def _finish(
    context: RunContext, payload: dict[str, Any], status: str, reason: str, started: float
) -> int:
    payload["status"] = status
    payload["reason"] = reason
    payload["finished_at"] = _now()
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(context.manifest_path, payload)
    print(f"{context.stage}: status={status} output={context.run_dir} reason={reason}")
    # Exit-code contract: a completed run succeeds, a real failure fails, and a
    # run that could not start is neither -- a scheduler must be able to tell the
    # three apart without parsing the manifest.
    return {"ok": 0, "fail": 1, "not_run": 2}[status]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = DEFAULT_CONFIGS[args.stage]
    try:
        cfg = n02_config.load_config(args.config)
    except n02_config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.seed is None:
        args.seed = int(cfg.seeds[0]) if cfg.seeds else 0
    if args.lr_scale <= 0:
        print(f"argument error: --lr-scale must be positive, got {args.lr_scale}", file=sys.stderr)
        return 2
    if args.updates is not None and args.updates <= 0:
        print(f"argument error: --updates must be positive, got {args.updates}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        "-m",
        "owac.training.n02_full",
        *(argv if argv is not None else sys.argv[1:]),
    ]
    context = build_run_context(args, cfg, command=command)
    context.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    payload = _base_payload(args, cfg, context)

    try:
        spec, checkpoint_spec = _resolve_specs(cfg, args.stage)
    except TrainingSpecError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    payload["mode_gate"] = mode_gate_payload(spec, cfg)
    if args.log_every:
        spec = dataclasses.replace(spec, log_every=int(args.log_every))
    if args.micro_batch is not None:
        if args.micro_batch <= 0:
            print("argument error: --micro-batch must be positive", file=sys.stderr)
            return 2
        if args.stage == "p1-pretrain":
            spec = dataclasses.replace(spec, micro_batch_frames=int(args.micro_batch))
        else:
            spec = dataclasses.replace(spec, micro_batch_episodes=int(args.micro_batch))
    if args.amp is not None:
        spec = dataclasses.replace(spec, amp=str(args.amp))
    if args.dev_eval_every is not None:
        if args.dev_eval_every <= 0:
            print(
                f"argument error: --dev-eval-every must be positive, got {args.dev_eval_every}",
                file=sys.stderr,
            )
            return 2
        if args.stage == "p1-pretrain":
            spec = dataclasses.replace(spec, dev_eval_every=int(args.dev_eval_every))
    if args.platforms:
        platforms = tuple(item.strip() for item in str(args.platforms).split(",") if item.strip())
        if not platforms:
            print("argument error: --platforms must not be empty", file=sys.stderr)
            return 2
        if args.stage == "p1-pretrain":
            spec = dataclasses.replace(spec, platforms=platforms)

    resume_payload = None
    if args.resume:
        try:
            resume_payload = load_checkpoint(args.resume)
        except TrainNotAvailable as exc:
            payload["missing"] = list(exc.missing or (str(args.resume),))
            return _finish(context, payload, "not_run", exc.reason, started)

    device = _device_for(args, cfg)
    if args.precision_check:
        platform = (args.platforms or "libero").split(",")[0].strip()
        try:
            receipt = run_precision_check(
                cfg,
                spec,
                context,
                args=args,
                device=device,
                platform=platform,
                micro_batch=args.precision_micro_batch,
            )
        except TrainNotAvailable as exc:
            payload["missing"] = list(exc.missing or (str(exc),))
            return _finish(context, payload, "not_run", exc.reason, started)
        write_json(context.run_dir / "precision_check.json", receipt)
        payload["precision_check"] = receipt
        status = "ok" if receipt["status"] == "ok" else "fail"
        return _finish(
            context,
            payload,
            status,
            "" if status == "ok" else "a reduced-precision mode exceeded its tolerance",
            started,
        )
    if args.stage == "p1-measurement" and args.arm in ("nre", "bnre"):
        try:
            outcome = run_ratio_arm(
                cfg,
                spec,
                context,
                arm=args.arm,
                updates=int(args.updates or 2000),
                device=device,
                base_checkpoint=str(args.resume or ""),
                batch=int(args.micro_batch or 16),
                log_every=int(args.log_every or 50),
                checkpoint_spec=checkpoint_spec,
            )
        except TrainNotAvailable as exc:
            payload["missing"] = list(exc.missing or (str(exc),))
            return _finish(context, payload, "not_run", exc.reason, started)
        except Exception as exc:  # noqa: BLE001 - a real failure must leave a receipt
            import traceback

            payload["traceback"] = traceback.format_exc()
            return _finish(context, payload, "fail", f"{type(exc).__name__}: {exc}", started)
        payload.update(
            {
                "updates": outcome["updates"],
                "updates_this_resume": outcome["updates_this_resume"],
                "resume_updates": outcome["resume_updates"],
                "checkpoint": outcome["checkpoint"],
                "partition": outcome.get("partition"),
                "loss_terms": summarise_terms(outcome["trainer"].state.history),
                "history_points": len(outcome["trainer"].state.history),
                "ablation_arm": outcome["arm_manifest"],
                "history": outcome["trainer"].state.history[-500:],
            }
        )
        write_json(context.metrics_path, payload)
        payload.pop("history", None)
        return _finish(context, payload, "ok", "", started)
    if args.stage == "p1-measurement" and args.arm != "main":
        # Bounded arms are tiny-tensor work; the same thread cap that made the NRE/BNRE arms
        # ~7x faster (7.8 -> 1.11 s/update) applies here too. Scheduling only.
        bound_torch_threads()
        if args.arm in ARM_MISSING_INTERFACES:
            payload["ablation_arm"] = arm_report(spec, args.arm, updates=args.updates)
            return _finish(
                context,
                payload,
                "not_run",
                f"arm {args.arm!r} needs an interface this harness does not have: "
                + ARM_MISSING_INTERFACES[args.arm],
                started,
            )
        spec = dataclasses.replace(spec, loss_weights=dict(ARM_LOSS_WEIGHTS[args.arm]))
    runner = run_pretrain if args.stage == "p1-pretrain" else run_measurement
    try:
        outcome = runner(
            cfg,
            spec,
            context,
            lr_scale=float(args.lr_scale),
            updates=args.updates,
            device=device,
            resume=resume_payload,
            checkpoint_spec=checkpoint_spec,
            **(
                {"synthetic_frames": bool(args.synthetic_frames)}
                if args.stage == "p1-pretrain"
                else {}
            ),
        )
    except TrainNotAvailable as exc:
        payload["missing"] = list(exc.missing or (str(exc),))
        return _finish(context, payload, "not_run", exc.reason, started)
    except Exception as exc:  # noqa: BLE001 - a real failure must leave a receipt
        import traceback

        payload["traceback"] = traceback.format_exc()
        return _finish(context, payload, "fail", f"{type(exc).__name__}: {exc}", started)

    trainer = outcome["trainer"]
    payload.update(
        {
            "updates": outcome["updates"],
            "updates_this_resume": outcome["updates_this_resume"],
            "resume_updates": outcome["resume_updates"],
            "checkpoint": outcome["checkpoint"],
            "sampler_position": trainer.sampler_state(),
            "partition": outcome.get("partition"),
            "coverage": outcome.get("coverage"),
            "loss_terms": summarise_terms(trainer.state.history),
            "history_points": len(trainer.state.history),
            "skipped_sources": outcome.get("skipped_sources", []),
            **(
                {
                    "ablation_arm": arm_report(
                        spec, str(getattr(args, "arm", "main")), updates=args.updates
                    )
                }
                if args.stage == "p1-measurement"
                else {}
            ),
        }
    )
    if args.stage == "p1-measurement":
        payload["unit_scale"] = unit_scale_payload(spec, trainer.state.history, cfg=cfg)
    if args.resume:
        payload["resume"] = {
            "from": str(args.resume),
            "resumed_from_stage": getattr(trainer, "resumed_from_stage", ""),
            "base_inheritance": getattr(trainer, "base_inheritance_report", {}),
            "resume_updates": outcome["resume_updates"],
            "new_updates": outcome["updates_this_resume"],
            "cumulative_updates": outcome["updates"],
        }
    payload["history"] = trainer.state.history[-500:]
    write_json(context.metrics_path, payload)
    payload.pop("history", None)
    return _finish(context, payload, "ok", "", started)


def unit_scale_payload(
    spec: MeasurementSpec,
    history: Sequence[Mapping[str, Any]],
    *,
    cfg: n02_config.N02FullConfig | None = None,
) -> dict[str, Any]:
    """Unit chain, typical proper-score magnitude and variance quantiles (freeze 3.8.5).

    The declared units of every source and the observed scale of the proper score are
    recorded so a later scale mismatch (for example a normalised-domain prediction
    scored against DN values) is visible in the receipt instead of being inferred.
    """
    values = []
    scale_suspect: Any = {"status": "not_run", "reason": "no check exposed scale_suspect"}
    variance_quantiles: dict[str, Any] = {
        "status": "not_run",
        "reason": "no check exposed variance quantiles in local_stats",
    }
    for record in history:
        for name, term in (record.get("terms") or {}).items():
            if name.startswith("raw_proper_prediction") and term.get("status") in (
                "ok",
                "detached",
            ):
                value = term.get("value")
                if isinstance(value, (int, float)) and value == value:
                    values.append(float(value))
    for record in reversed(history):
        for term in (record.get("terms") or {}).values():
            if "scale_suspect" in term:
                scale_suspect = {
                    "status": "ok",
                    "value": bool(term["scale_suspect"]),
                    "source": "T2 score_diagnostics",
                }
                break
        if scale_suspect.get("status") == "ok":
            break
    for record in reversed(history):
        for term in (record.get("terms") or {}).values():
            stats = term.get("variance_quantiles")
            if isinstance(stats, Mapping):
                variance_quantiles = {"status": "ok", **{str(k): v for k, v in stats.items()}}
                break
        if variance_quantiles["status"] == "ok":
            break
    domain = str(model_section(cfg).get("prediction_domain", "normalized")) if cfg else "normalized"
    affines: list[dict[str, Any]] = []
    sensors = spec.synthetic_sensors or ()
    for sensor in sensors:
        if not isinstance(sensor, Mapping):
            continue
        black = float(sensor.get("black_level", 0.0))
        white = float(sensor.get("white_level", 0.0))
        affines.append(
            {
                "arm": "synthetic",
                "camera_id": str(sensor.get("camera_id", "")),
                "prediction_domain": domain,
                "scale": (white - black) if white > black else 1.0,
                "offset": black if white > black else 0.0,
                "units": spec.units or _synthetic_units(spec),
                "calibration_version": "declared SensorSpec black/white",
                "source": "frame calibration metadata (T1 normalization inverted)",
            }
        )
    if "real" in spec.sources:
        affines.append(
            {
                "arm": "real",
                "camera_id": "*",
                "prediction_domain": "units",
                "scale": 1.0,
                "offset": 0.0,
                "units": spec.real_units,
                "calibration_version": "",
                "source": (
                    "the RawVLA cache declares no calibration metadata, so the prediction is "
                    "already in the declared query units (identity); density is scored as "
                    + ("proper_nll" if spec.real_density_verified else "empirical_energy")
                ),
            }
        )
    sensor_section = dict(model_section(cfg)) if cfg is not None else {}
    sensor_parameters: dict[str, Any] = {
        "per_camera": dict(sensor_section.get("per_camera_sensor", {}) or {}),
        "dn_mean_convention": str(sensor_section.get("dn_mean_convention", "")),
        "per_camera_note": str(sensor_section.get("per_camera_note", "")),
        "run_level": {
            key: sensor_section.get(key)
            for key in (
                "quantisation_step",
                "saturation_level",
                "floor_level",
                "shot_noise_scale",
                "read_noise_signal",
                "noise_scale",
                "prediction_domain",
            )
        },
    }
    try:
        model = build_sensor_model(cfg, spec.units or _synthetic_units(spec), density_verified=True)
        semantics = model.unit_semantics()
        sensor_parameters["unit_semantics"] = dict(semantics)
    except Exception as exc:  # noqa: BLE001 - the block reports the absence, never invents it
        sensor_parameters["unit_semantics"] = {
            "status": "not_run",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {
        "sensor_parameters": sensor_parameters,
        "scale_suspect": scale_suspect,
        "affines": affines,
        "synthetic_units": spec.units or _synthetic_units(spec),
        "real_units": spec.real_units,
        "real_density": ("proper_nll" if spec.real_density_verified else "empirical_energy"),
        "ce_temperature": float(spec.ce_temperature),
        "proper_score_magnitude": {
            "updates_recorded": len(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "order_of_magnitude": (
                None if not values else float(f"{max(abs(v) for v in values):.0e}")
            ),
        },
        "variance_quantiles": variance_quantiles,
    }


def summarise_terms(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-term summary over the run history (activation counts, ranges)."""
    if not history:
        return {}
    names = sorted({name for record in history for name in (record.get("terms") or {})})
    summary: dict[str, Any] = {}
    for name in names:
        records = [
            record["terms"][name] for record in history if name in (record.get("terms") or {})
        ]
        values = [
            float(record["value"])
            for record in records
            if record.get("status") in ("ok", "detached") and record.get("value") is not None
        ]
        summary[name] = {
            "updates_present": len(values),
            "updates_total": len(history),
            "statuses": sorted({str(record.get("status")) for record in records}),
            "first_value": values[0] if values else None,
            "last_value": values[-1] if values else None,
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "activation_count_total": int(
                sum(int(record.get("activation_count") or 0) for record in records)
            ),
            "labels": sorted(
                {
                    str(record.get("label_source"))
                    for record in records
                    if record.get("label_source")
                }
            ),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())

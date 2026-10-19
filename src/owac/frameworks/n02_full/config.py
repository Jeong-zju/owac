"""Versioned configuration for the N02 full (v1.1) development line.

The config file is a flat JSON object whose keys are the dataclass fields of
:class:`N02FullConfig`. A missing key keeps its declared default; an unknown
top-level key or a type mismatch is an error that names the offending key.
Nested objects (``data``, ``runtime``, ``profile``, ``paths``, ``partitions``)
are validated for the types of their known keys but stay open for extra fields,
because they carry environment-specific notes and probe results that the
framework does not own.

``config_hash`` is the canonical hash of the resolved config, so it changes when
any declared field changes and is stable across dict ordering and JSON
formatting. It is the hash recorded in stage decisions and run manifests.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CONFIG_VERSION",
    "P1_CONFIG_VERSION",
    "ConfigError",
    "N02FullConfig",
    "as_dict",
    "config_hash",
    "load_config",
    "validate_config",
]

#: Version of this config schema, not of the model. Stage decisions record it.
CONFIG_VERSION = "n02-full-p0-v1"

#: Schema version once the P1 ``training``/``evaluation`` sections exist. The P0
#: constant above is frozen because P0 receipts and stage decisions record it.
P1_CONFIG_VERSION = "n02-full-p1-v1"

_RUNTIME_DTYPES = ("float32", "float16", "bfloat16")

_PARTITION_NAMES = ("train", "dev", "calibration")

#: Sections the P1 training/evaluation entry points read. They are validated for
#: structure here and resolved into strict dataclasses by their owning module, so
#: a typo in a nested hyper-parameter name fails there with the full key path.
_TRAINING_SECTIONS = (
    "pretrain",
    "measurement",
    "selection",
    "data",
    "checkpoint",
    "instrumentation",
)
_EVALUATION_SECTIONS = ("scenarios", "thresholds", "artifacts")


class ConfigError(ValueError, TypeError):
    """A config file problem: unknown key, wrong type, or unreadable JSON.

    It derives from both :class:`ValueError` and :class:`TypeError` so callers
    may catch either without losing the more precise class.
    """


def _default_partitions() -> dict:
    return {"train": 0.8, "dev": 0.1, "calibration": 0.1}


def _default_data() -> dict:
    return {
        "cache_root": None,
        "dataset_format": None,
        "protocol_verified": False,
        "notes": "",
    }


def _default_runtime() -> dict:
    return {
        "dtype": "float32",
        "device": "cpu",
        "amp": False,
        "acceptance_budgets": {
            "decision_p95_ms_per_episode": 200,
            "persistent_state_bytes_per_episode": 268435456,
            "scope": (
                "development budget; full integrated decision and retained state must be "
                "measured at P1/P2, no full-model profile claimed"
            ),
            "frozen_date": "2026-10-14",
        },
    }


def _default_profile() -> dict:
    return {
        "updates": 200,
        "warmup_updates": 20,
        "batch_size": 2,
        "sequence_length": 8,
        "operation": "train_step",
    }


def _default_training() -> dict:
    """P1 training sections; empty means "the owner module's declared default".

    The P1 entry points resolve each section into a strict dataclass, so an empty
    section keeps the frozen P1 starting values from the stage prompt and a
    partially specified section overrides only the keys it names.
    """
    return {name: {} for name in _TRAINING_SECTIONS}


def _default_evaluation() -> dict:
    return {name: {} for name in _EVALUATION_SECTIONS}


@dataclass(frozen=True, slots=True)
class N02FullConfig:
    """Resolved configuration. Frozen: a run derives a new config instead.

    ``partitions`` sums to 1 over train/dev/calibration; the sealed test set is
    a separately tracked manifest and is never a partition weight here.
    ``data`` records the audited data root and protocol state, ``runtime`` the
    numerical mode, ``profile`` the profiling budget and ``paths`` the output
    root. Those five dicts are carried through as resolved values; their content
    is validated by :func:`validate_config` where the contract fixes it.
    """

    version: str = "n02-full-v1.1"
    seeds: tuple[int, ...] = (1103, 2207, 3301)
    worlds: int = 8
    slots: int = 16
    slot_dim: int = 512
    depths: tuple[int, ...] = (3, 3, 27, 3)
    dims: tuple[int, ...] = (128, 256, 512, 1024)
    input_size: int = 256
    scales: tuple[str, ...] = ("stride8", "stride16")
    partitions: dict = field(default_factory=_default_partitions)
    data: dict = field(default_factory=_default_data)
    runtime: dict = field(default_factory=_default_runtime)
    profile: dict = field(default_factory=_default_profile)
    paths: dict = field(default_factory=lambda: {"output_root": "outputs/n02-full"})
    training: dict = field(default_factory=_default_training)
    evaluation: dict = field(default_factory=_default_evaluation)


_FIELD_NAMES = tuple(item.name for item in dataclasses.fields(N02FullConfig))

_TUPLE_FIELDS = {
    "seeds": "int",
    "depths": "int",
    "dims": "int",
    "scales": "str",
}


def _expect_str(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"config key {key!r}: expected a string, got {type(value).__name__}")
    return value


def _expect_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"config key {key!r}: expected a boolean, got {type(value).__name__}")
    return value


def _expect_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"config key {key!r}: expected an integer, got {type(value).__name__}")
    return value


def _expect_number(value: Any, key: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"config key {key!r}: expected a number, got {type(value).__name__}")
    return value


def _expect_scalar_tuple(value: Any, key: str, kind: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"config key {key!r}: expected a list of {kind}s, got {type(value).__name__}"
        )
    items = []
    for index, item in enumerate(value):
        if kind == "int":
            items.append(_expect_int(item, f"{key}[{index}]"))
        else:
            items.append(_expect_str(item, f"{key}[{index}]"))
    return tuple(items)


def _expect_mapping(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"config key {key!r}: expected an object, got {type(value).__name__}")
    return copy.deepcopy(value)


def _check_partitions(value: Any, key: str) -> dict:
    mapping = _expect_mapping(value, key)
    for name, weight in mapping.items():
        if not isinstance(name, str):
            raise ConfigError(f"config key {key!r}: partition names must be strings")
        _expect_number(weight, f"{key}.{name}")
    return mapping


def _check_data(value: Any, key: str) -> dict:
    mapping = _expect_mapping(value, key)
    if "cache_root" in mapping and mapping["cache_root"] is not None:
        _expect_str(mapping["cache_root"], f"{key}.cache_root")
    if "dataset_format" in mapping and mapping["dataset_format"] is not None:
        _expect_str(mapping["dataset_format"], f"{key}.dataset_format")
    if "protocol_verified" in mapping:
        _expect_bool(mapping["protocol_verified"], f"{key}.protocol_verified")
    if "notes" in mapping:
        _expect_str(mapping["notes"], f"{key}.notes")
    return mapping


def _check_runtime(value: Any, key: str) -> dict:
    mapping = _expect_mapping(value, key)
    if "dtype" in mapping:
        _expect_str(mapping["dtype"], f"{key}.dtype")
    if "device" in mapping:
        _expect_str(mapping["device"], f"{key}.device")
    if "amp" in mapping:
        _expect_bool(mapping["amp"], f"{key}.amp")
    return mapping


def _check_profile(value: Any, key: str) -> dict:
    mapping = _expect_mapping(value, key)
    for name in ("updates", "warmup_updates", "batch_size", "sequence_length"):
        if name in mapping:
            _expect_int(mapping[name], f"{key}.{name}")
    return mapping


def _check_paths(value: Any, key: str) -> dict:
    mapping = _expect_mapping(value, key)
    if "output_root" in mapping:
        _expect_str(mapping["output_root"], f"{key}.output_root")
    return mapping


def _check_sections(value: Any, key: str, known: tuple[str, ...]) -> dict:
    """A named-section map: every known section, when present, is an object.

    It stays open for extra keys (a run may record derived notes there), but a
    known section that is not an object is rejected here with its key path so the
    failure is attributed before the owner module resolves the section.
    """
    mapping = _expect_mapping(value, key)
    for name in known:
        if name in mapping and not isinstance(mapping[name], dict):
            raise ConfigError(
                f"config key {key}.{name}: expected an object, got {type(mapping[name]).__name__}"
            )
    return mapping


_DICT_FIELDS = {
    "partitions": _check_partitions,
    "data": _check_data,
    "runtime": _check_runtime,
    "profile": _check_profile,
    "paths": _check_paths,
    "training": lambda value, key: _check_sections(value, key, _TRAINING_SECTIONS),
    "evaluation": lambda value, key: _check_sections(value, key, _EVALUATION_SECTIONS),
}


def load_config(path: str | Path) -> N02FullConfig:
    """Load and validate one config JSON file.

    Missing keys keep the declared defaults. Unknown top-level keys, malformed
    JSON and type mismatches raise :class:`ConfigError` with the key name in the
    message. Semantic problems that do not prevent construction (for example a
    partition table that does not sum to 1) are reported by
    :func:`validate_config`, not here.
    """
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file {target} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(
            f"config file {target} must contain a JSON object, got {type(payload).__name__}"
        )
    unknown = sorted(set(payload) - set(_FIELD_NAMES))
    if unknown:
        raise ConfigError(f"unknown config key(s): {', '.join(repr(name) for name in unknown)}")

    values: dict[str, Any] = {}
    for name, value in payload.items():
        if name in _TUPLE_FIELDS:
            values[name] = _expect_scalar_tuple(value, name, _TUPLE_FIELDS[name])
        elif name in _DICT_FIELDS:
            values[name] = _DICT_FIELDS[name](value, name)
        elif name == "version":
            values[name] = _expect_str(value, name)
        else:
            values[name] = _expect_int(value, name)
    return N02FullConfig(**values)


def as_dict(cfg: N02FullConfig) -> dict:
    """Plain, JSON-ready copy of the config; tuples become lists."""
    payload: dict[str, Any] = {}
    for name in _FIELD_NAMES:
        value = getattr(cfg, name)
        if isinstance(value, tuple):
            payload[name] = list(value)
        elif isinstance(value, dict):
            payload[name] = copy.deepcopy(value)
        else:
            payload[name] = value
    return payload


def canonical_json(payload: Any) -> str:
    """Canonical JSON text used for every hash in this module."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: N02FullConfig) -> str:
    """Stable sha256 over the resolved config (canonical JSON, sorted keys)."""
    return hashlib.sha256(canonical_json(as_dict(cfg)).encode("utf-8")).hexdigest()


def validate_config(cfg: N02FullConfig) -> list[str]:
    """Report semantic problems as strings; never raises, never repairs.

    An empty list means the config is internally consistent. Type problems are
    reported here too so a directly constructed (not file-loaded) config is
    checked as well.
    """
    issues: list[str] = []

    if not cfg.seeds:
        issues.append("seeds: must be non-empty")
    elif not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in cfg.seeds):
        issues.append("seeds: every seed must be an integer")

    for name in ("worlds", "slots", "slot_dim", "input_size"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int):
            issues.append(f"{name}: expected an integer, got {type(value).__name__}")
        elif value <= 0:
            issues.append(f"{name}: must be positive, got {value}")

    if len(cfg.depths) != len(cfg.dims):
        issues.append(
            f"depths/dims: must have equal length, got {len(cfg.depths)} and {len(cfg.dims)}"
        )
    for name in ("depths", "dims"):
        values = getattr(cfg, name)
        if not values:
            issues.append(f"{name}: must be non-empty")
        elif not all(isinstance(item, int) and not isinstance(item, bool) for item in values):
            issues.append(f"{name}: every entry must be an integer")
        elif any(item <= 0 for item in values):
            issues.append(f"{name}: every entry must be positive, got {tuple(values)!r}")

    if not cfg.scales:
        issues.append("scales: must be non-empty")
    else:
        for scale in ("stride8", "stride16"):
            if scale not in cfg.scales:
                issues.append(f"scales: must contain {scale!r}, got {tuple(cfg.scales)!r}")

    if not isinstance(cfg.partitions, dict) or not cfg.partitions:
        issues.append("partitions: must be a non-empty object")
    else:
        missing = [name for name in _PARTITION_NAMES if name not in cfg.partitions]
        if missing:
            issues.append(f"partitions: missing partition(s) {', '.join(missing)}")
        values = list(cfg.partitions.values())
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in values
        ):
            issues.append("partitions: every weight must be a number")
        else:
            total = float(sum(values))
            if abs(total - 1.0) > 1e-6:
                issues.append(f"partitions: weights must sum to 1, got {total!r}")
            if any(float(value) < 0.0 for value in values):
                issues.append("partitions: weights must be non-negative")

    dtype = cfg.runtime.get("dtype") if isinstance(cfg.runtime, dict) else None
    if dtype not in _RUNTIME_DTYPES:
        allowed = ", ".join(_RUNTIME_DTYPES)
        issues.append(f"runtime.dtype: must be one of {allowed}, got {dtype!r}")

    for field_name, known in (
        ("training", _TRAINING_SECTIONS),
        ("evaluation", _EVALUATION_SECTIONS),
    ):
        sections = getattr(cfg, field_name, None)
        if not isinstance(sections, dict):
            issues.append(f"{field_name}: must be an object, got {type(sections).__name__}")
            continue
        for name in known:
            entry = sections.get(name)
            if entry is None:
                continue
            if not isinstance(entry, dict):
                issues.append(f"{field_name}.{name}: must be an object, got {type(entry).__name__}")
                continue
            if field_name != "training" or name not in ("pretrain", "measurement"):
                continue
            for key in ("lr", "weight_decay", "warmup_fraction", "mask_ratio"):
                if key not in entry:
                    continue
                value = entry[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    issues.append(
                        f"{field_name}.{name}.{key}: expected a number, got {type(value).__name__}"
                    )
                elif value < 0:
                    issues.append(f"{field_name}.{name}.{key}: must be non-negative, got {value}")
            if "updates" in entry:
                value = entry["updates"]
                if isinstance(value, bool) or not isinstance(value, int):
                    issues.append(
                        f"{field_name}.{name}.updates: expected an integer, "
                        f"got {type(value).__name__}"
                    )
                elif value <= 0:
                    issues.append(f"{field_name}.{name}.updates: must be positive, got {value}")

    return issues

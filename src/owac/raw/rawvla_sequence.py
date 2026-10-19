"""RawVLA NPZ sequence loader for the N01 measurement path.

Implements the "两种输入分别实现" step: LIBERO ``*_raw_uint8`` pseudo-RAW and
RoboTwin ``*_raw_float32`` HDR-derived RAW become causal :class:`MeasurementWindow`
measurement tensors that keep per-frame episode/camera/frame ids, an explicit
``new_measurement`` flag for repeated padding frames and a separate action mask for
padded action tails.  Action indexing and statistics follow the author loader
(``starVLA/dataloader/rawvla_npz_datasets.py``, commit 57d508c) while its PIL image
output and RGB chroma target are dropped.  LIBERO uint8 becomes FP32 and is divided
by 255; RoboTwin float32 is passed through untouched (recorded author samples top out
at 0.433, and the author chain never rescales a non-uint8 burst).  The server snapshot
``/workspace/owac/rawvla`` is absent on this machine, so a missing snapshot raises
:class:`MissingRawVLAAssetError` naming the exact paths instead of inventing data; the
tests use genuine synthetic NPZ fixtures written with the author key names.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised by the environment, not by the tests
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "owac.raw.rawvla_sequence needs the 'encoder' extra (numpy + torch): "
        "uv sync --extra encoder"
    ) from exc

from owac.core.encoder_contract import EmbodimentContext, MeasurementFrame
from owac.raw.cfa import CfaPlaneStack, NoiseModel

__all__ = [
    "DEFAULT_RAWVLA_ROOT",
    "LIBERO_NOISE_MODEL",
    "LIBERO_PROTOCOL",
    "ROBOTWIN_PROTOCOL",
    "MissingRawVLAAssetError",
    "AssetRequirement",
    "MeasurementWindow",
    "MeasurementWindowBatch",
    "RawVLAProtocol",
    "RawVLASequenceDataset",
    "SequenceSample",
    "WindowPlan",
    "assign_side",
    "build_split_manifest",
    "collate_measurement_windows",
    "collate_sequence_samples",
    "default_model_root",
    "default_rawvla_root",
    "load_split_manifest",
    "measurement_window_from_cfa",
    "rawvla_asset_requirements",
    "recorded_range_evidence",
    "source_group",
    "write_split_manifest",
]

DEFAULT_RAWVLA_ROOT = Path("/workspace/owac/rawvla")

#: Author LIBERO RAW chain (``rawvla_bench/raw.py``, ``DEFAULT_SENSOR_NOISE``).
#: The generator adds ``4e-4*raw + 1e-5`` to the normalised raw10 value, which is
#: exactly ``x`` after the uint8 view is divided by 255, so the same numbers apply
#: to the loaded measurement unit.  8-bit quantisation adds about ``(1/255)**2/12``
#: of variance that this model does not include.
LIBERO_NOISE_MODEL = NoiseModel(
    shot=4.0e-4,
    read_var=1.0e-5,
    units="pseudo_raw_linear_view_uint8_over_255",
    fitted=False,
    estimator=False,
    evidence={
        "source": (
            "rawvla/experiments/simulation/libero/rawvla_bench/raw.py "
            "DEFAULT_SENSOR_NOISE (shot_noise=4e-4, read_noise=1e-5)"
        ),
        "note": (
            "noise was applied in the normalised raw10 domain, which equals x after the "
            "uint8 view is divided by 255; quantisation variance is excluded"
        ),
    },
)

TIMESTAMP_KEYS: tuple[str, ...] = (
    "timestamps",
    "timestamps_seconds",
    "timestamp",
    "time_stamps",
    "observation_timestamps",
    "time",
)

IDENTITY_METADATA_KEYS: tuple[str, ...] = (
    "source_group",
    "capture_session",
    "session_id",
    "acquisition_id",
)

SUPPORTED_NORMALIZATIONS: tuple[str, ...] = (
    "pi05_quantile",
    "quantile",
    "zscore",
    "pi0_zscore",
    "starvla_minmax",
    "none",
)

VARIANT_DIR = re.compile(r"^variant_\d+$")
VARIANT_SUFFIX = re.compile(r"^(?P<base>.+)_variant_\d+$")
LIGHTING_SUFFIX = re.compile(r"^(?P<base>.+)_lighting_[A-Za-z0-9]+$")


@dataclass(frozen=True)
class RawVLAProtocol:
    """Author cache protocol for one RawVLA benchmark family."""

    name: str
    aliases: tuple[str, ...]
    camera_keys: tuple[str, ...]
    camera_dtype: str
    raw_scaling: str
    raw_unit: str
    raw_range_evidence: str
    state_source_key: str
    state_dim: int
    state_kind: str
    state_note: str
    proprio_available: bool
    action_dim: int
    action_offset: int
    action_horizon: int
    observation_stride: int
    observation_stride_verified: bool
    normalization: str

    def camera_id(self, key: str) -> str:
        """Camera id for a stored key, e.g. ``head_camera_raw_float32`` -> ``head_camera``."""
        for suffix in ("_raw_uint8", "_raw_float32", "_raw"):
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key


LIBERO_PROTOCOL = RawVLAProtocol(
    name="libero",
    aliases=("libero",),
    camera_keys=("agentview_raw_uint8", "wrist_raw_uint8"),
    camera_dtype="uint8",
    raw_scaling="uint8_to_float32_over_255",
    raw_unit="pseudo_raw_linear_view_uint8_over_255",
    raw_range_evidence=(
        "author chain rawvla_bench/raw.py: rgb8 -> exposed raw10 -> noise in the normalised "
        "[0,1] domain -> round(raw10*255/1023) uint8 view, so uint8 stores a linear RAW10 view "
        "and /255 reproduces the author's own burst scaling (the author loader divides only "
        "when the stored dtype is uint8). Recorded sample "
        "data/libero/raw/libero_spatial_no_noops_1.0.0_lerobot/episode_000018/variant_00.npz "
        "(RawVLA-Bench 8586c59, dataset-check.json): uint8 min 0 max 8 -> x in [0, 0.0314]."
    ),
    state_source_key="robot_states",
    state_dim=8,
    state_kind="measured_proprio",
    state_note=(
        "robot_states[t] is the 8-dim proprio field. The 92-dim 'states' array is simulator "
        "state and is never used as a policy input."
    ),
    proprio_available=True,
    action_dim=7,
    action_offset=0,
    action_horizon=10,
    observation_stride=5,
    observation_stride_verified=True,
    normalization="pi05_quantile",
)

ROBOTWIN_PROTOCOL = RawVLAProtocol(
    name="robotwin2",
    aliases=("robotwin2", "robotwin"),
    camera_keys=(
        "head_camera_raw_float32",
        "left_camera_raw_float32",
        "right_camera_raw_float32",
    ),
    camera_dtype="floating",
    raw_scaling="author_float32_passthrough",
    raw_unit="hdr_derived_raw_float32_author_units",
    raw_range_evidence=(
        "author cache stores head/left/right_camera_raw_float32 as HDR-derived RAW float32 in "
        "its own units; the author loader divides only uint8 bursts and never rescales float32. "
        "Recorded sample data/robotwin2/raw/grab_roller/episode_000018.npz (RawVLA-Bench "
        "8586c59, dataset-check.json): head max 0.4332, left max 0.3744, right max 0.3738, "
        "min 0.0 (right min 2.6e-4). Dividing by 255 would push the whole scene below 1.7e-3, "
        "and a per-frame max scaling would destroy the lighting-domain information, so OWAC "
        "passes these arrays through and applies only a calibration declared in the metadata."
    ),
    state_source_key="actions",
    state_dim=14,
    state_kind="cached_joint_target_channel",
    state_note=(
        "actions[t][:14] is the cached 14-dim joint target channel that the author reuses as "
        "the current state; the 16-dim robot_states is an end-pose representation and is never "
        "used as the joint state. The cache records commands, not measured executed actions."
    ),
    proprio_available=False,
    action_dim=14,
    action_offset=1,
    action_horizon=50,
    observation_stride=1,
    observation_stride_verified=False,
    normalization="zscore",
)

PROTOCOLS: dict[str, RawVLAProtocol] = {}
for _protocol in (LIBERO_PROTOCOL, ROBOTWIN_PROTOCOL):
    for _alias in _protocol.aliases:
        PROTOCOLS[_alias] = _protocol


def resolve_protocol(dataset_format: str) -> RawVLAProtocol:
    key = str(dataset_format).lower()
    if key not in PROTOCOLS:
        raise ValueError(
            f"unknown RawVLA dataset_format {dataset_format!r}; expected one of {sorted(PROTOCOLS)}"
        )
    return PROTOCOLS[key]


# --------------------------------------------------------------------------------------
# Asset discovery and readable missing-asset errors
# --------------------------------------------------------------------------------------


def default_rawvla_root() -> Path:
    """Server snapshot root; ``RAWVLA_ROOT`` overrides the built-in default."""
    override = os.environ.get("RAWVLA_ROOT", "").strip()
    return Path(override) if override else DEFAULT_RAWVLA_ROOT


def default_model_root() -> Path:
    """Checkpoint/normalisation-statistics root; ``RAWVLA_MODEL_ROOT`` overrides it."""
    override = os.environ.get("RAWVLA_MODEL_ROOT", "").strip()
    return Path(override) if override else default_rawvla_root() / "checkpoints"


def default_raw_cache_root(root: str | Path, dataset_format: str) -> Path:
    """Raw NPZ cache directory inside the downloaded RawVLA-Bench snapshot."""
    protocol = resolve_protocol(dataset_format)
    return Path(root) / "data" / protocol.name / "raw"


def default_rgb_cache_root(root: str | Path, dataset_format: str) -> Path:
    """Paired RGB chroma cache; recorded for provenance only, never loaded by OWAC."""
    protocol = resolve_protocol(dataset_format)
    return Path(root) / "data" / protocol.name / "rgb"


def default_stats_path(root: str | Path, dataset_format: str) -> Path:
    """Author normalisation statistics file declared by the pinned author configs."""
    protocol = resolve_protocol(dataset_format)
    if protocol.name == "libero":
        return (
            Path(root)
            / "openpi_converted_protocol"
            / "pi05_libero_starvla"
            / "bfloat16"
            / "assets"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        )
    return default_model_root() / "pi05_robotwin" / "policy_norm_stats.json"


@dataclass(frozen=True)
class AssetRequirement:
    """One path the loader needs, with its provenance so an error stays actionable."""

    kind: str
    path: Path
    required: bool
    env_var: str | None
    note: str
    satisfied: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "required": bool(self.required),
            "satisfied": bool(self.satisfied),
            "env_var": self.env_var,
            "note": self.note,
        }


def rawvla_asset_requirements(
    *,
    root: str | Path | None = None,
    dataset_format: str = "libero",
    raw_cache_root: str | Path | None = None,
    stats_path: str | Path | None = None,
    rgb_cache_root: str | Path | None = None,
    requires_stats: bool = True,
) -> list[AssetRequirement]:
    """Describe every asset the loader needs without raising, in report order."""
    protocol = resolve_protocol(dataset_format)
    root_path = Path(root) if root is not None else default_rawvla_root()
    cache = (
        Path(raw_cache_root)
        if raw_cache_root is not None
        else default_raw_cache_root(root_path, protocol.name)
    )
    stats = (
        Path(stats_path) if stats_path is not None else default_stats_path(root_path, protocol.name)
    )
    rgb = (
        Path(rgb_cache_root)
        if rgb_cache_root is not None
        else default_rgb_cache_root(root_path, protocol.name)
    )
    root_env = "RAWVLA_ROOT" if root is None else None
    stats_env = (
        "RAWVLA_MODEL_ROOT" if protocol.name != "libero" and stats_path is None else root_env
    )
    return [
        AssetRequirement(
            kind="root",
            path=root_path,
            required=True,
            env_var=root_env,
            note="RawVLA server snapshot root holding the RawVLA-Bench download",
            satisfied=root_path.is_dir(),
        ),
        AssetRequirement(
            kind="raw_cache",
            path=cache,
            required=True,
            env_var=root_env,
            note=(
                f"{protocol.name} raw NPZ cache; expected author keys "
                f"{', '.join(protocol.camera_keys)}"
            ),
            satisfied=cache.is_dir() and bool(sorted(cache.glob("**/*.npz"))),
        ),
        AssetRequirement(
            kind="stats",
            path=stats,
            required=bool(requires_stats),
            env_var=stats_env,
            note=f"author normalisation statistics ({protocol.normalization})",
            satisfied=stats.is_file(),
        ),
        AssetRequirement(
            kind="rgb_cache",
            path=rgb,
            required=False,
            env_var=root_env,
            note=(
                "author paired RGB chroma cache; OWAC never loads it because the main path "
                "produces no RGB target"
            ),
            satisfied=rgb.is_dir(),
        ),
    ]


class MissingRawVLAAssetError(FileNotFoundError):
    """Raised when the RawVLA snapshot or its statistics are not present locally."""

    def __init__(
        self,
        requirements: Sequence[AssetRequirement],
        *,
        dataset_format: str,
        root: str | Path,
    ) -> None:
        self.requirements = list(requirements)
        self.missing = [item for item in self.requirements if item.required and not item.satisfied]
        self.dataset_format = dataset_format
        self.root = Path(root)
        protocol = resolve_protocol(dataset_format)
        lines = [
            f"RawVLA assets for dataset_format={protocol.name!r} are not available "
            f"(root={self.root}).",
            "No RawVLA data was loaded; nothing below is reported as measured.",
            "Missing required assets:",
        ]
        for item in self.missing:
            env = f" (set {item.env_var} or pass the path explicitly)" if item.env_var else ""
            lines.append(f"  [{item.kind}] {item.path}{env} - {item.note}")
        lines.append(
            f"Expected layout: <root>/data/{protocol.name}/raw/**/*.npz with author camera "
            f"keys {', '.join(protocol.camera_keys)}."
        )
        lines.append(
            "Point root= at the snapshot (or set RAWVLA_ROOT / RAWVLA_MODEL_ROOT) to load real "
            "data. The loader also accepts small synthetic NPZ files written with the same key "
            "names; those must declare metadata_json['synthetic_fixture']=true and are never "
            "reported as real RawVLA data."
        )
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------------------------
# Deterministic, source-aware splits
# --------------------------------------------------------------------------------------


def _format_aliases(dataset_format: str) -> tuple[str, ...]:
    return resolve_protocol(dataset_format).aliases


def source_group(relative_path: str | Path, *, dataset_format: str = "libero") -> str:
    """Source unit of one cache file: RGB/RAW/lighting variants share one group.

    The ``data/<format>/{raw,rgb}/`` prefix is dropped, a trailing ``variant_NN``
    directory is dropped and a trailing ``_variant_NN``/``_lighting_<domain>``
    suffix is stripped, so paired RGB, RAW and lighting-derivative files of one
    trajectory arrive at the same key and can never be split apart.
    """
    posix = Path(str(relative_path)).as_posix()
    while posix.startswith("./"):
        posix = posix[2:]
    prefixes: list[str] = []
    for alias in _format_aliases(dataset_format):
        for kind in ("raw", "rgb"):
            prefixes.append(f"data/{alias}/{kind}/")
            prefixes.append(f"{alias}/{kind}/")
    for kind in ("raw", "rgb"):
        prefixes.append(f"{kind}/")
    for prefix in prefixes:
        if posix.startswith(prefix):
            posix = posix[len(prefix) :]
            break
    parts = [part for part in posix.split("/") if part not in ("", ".")]
    if parts and VARIANT_DIR.match(parts[-1]):
        parts.pop()
    if parts:
        stem = parts[-1]
        if stem.endswith(".npz"):
            stem = stem[: -len(".npz")]
        for _ in range(4):
            match = None
            for pattern in (VARIANT_SUFFIX, LIGHTING_SUFFIX):
                match = pattern.match(stem)
                if match:
                    break
            if match is None:
                break
            stem = match.group("base")
        if stem == "" or VARIANT_DIR.match(stem):
            # the whole trailing component was a variant/lighting tag
            parts.pop()
        else:
            parts[-1] = stem
    return "/".join(parts) if parts else posix


def assign_side(
    group: str,
    *,
    split_version: str = "n01-rawvla-split-v1",
    train_fraction: float = 0.8,
) -> str:
    """Deterministic side of one source group: ``sha256(split_version|group)``."""
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError(f"train_fraction must be inside (0, 1), got {train_fraction}")
    digest = hashlib.sha256(f"{split_version}|{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "train" if value < float(train_fraction) else "validation"


def build_split_manifest(
    paths: Sequence[str | Path],
    *,
    dataset_format: str = "libero",
    relative_to: str | Path | None = None,
    train_fraction: float = 0.8,
    split_version: str = "n01-rawvla-split-v1",
    metadata_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a reproducible train/validation split over source groups.

    Identical source groups always land on the same side, so paired RGB caches and
    lighting variants of one trajectory cannot leak across the split.  The
    official closed-loop manifest is never consulted or merged here.
    """
    protocol = resolve_protocol(dataset_format)
    base = Path(relative_to) if relative_to is not None else None
    groups: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if base is not None:
            try:
                relative = path.relative_to(base)
            except ValueError:
                relative = path
        else:
            relative = path
        metadata = (metadata_by_path or {}).get(str(path), {})
        identity_field = "relative_path"
        group = source_group(relative, dataset_format=protocol.name)
        for key in IDENTITY_METADATA_KEYS:
            value = metadata.get(key)
            if value not in (None, ""):
                group = f"{protocol.name}:{value}"
                identity_field = f"metadata.{key}"
                break
        entry = groups.setdefault(
            group,
            {
                "group": group,
                "side": assign_side(
                    group, split_version=split_version, train_fraction=train_fraction
                ),
                "identity_field": identity_field,
                "files": [],
                "lighting_domains": [],
            },
        )
        entry["files"].append(str(path))
        domain = metadata.get("lighting_domain")
        if domain not in (None, "") and domain not in entry["lighting_domains"]:
            entry["lighting_domains"].append(domain)
    ordered = sorted(groups.values(), key=lambda item: item["group"])
    for entry in ordered:
        entry["files"] = sorted(entry["files"])
        entry["lighting_domains"] = sorted(entry["lighting_domains"])
    sides: dict[str, list[str]] = {"train": [], "validation": []}
    for entry in ordered:
        sides[entry["side"]].extend(entry["files"])
    sides = {side: sorted(files) for side, files in sides.items()}
    counts = {
        "files": sum(len(files) for files in sides.values()),
        "groups": len(ordered),
        "train_files": len(sides["train"]),
        "validation_files": len(sides["validation"]),
        "train_groups": sum(1 for entry in ordered if entry["side"] == "train"),
        "validation_groups": sum(1 for entry in ordered if entry["side"] == "validation"),
    }
    notes = [
        "rule: sha256(split_version + '|' + source_group) -> u64/2**64 < train_fraction",
        "paired RGB/RAW files and lighting variants of one source share a group by construction",
        "official RawVLA closed-loop manifests are kept separate and never mixed into this split",
    ]
    if counts["validation_files"] == 0 or counts["train_files"] == 0:
        notes.append(
            "warning: one side is empty for this inventory; adjust train_fraction or "
            "split_version before using it for model selection"
        )
    return {
        "split_version": split_version,
        "dataset_format": protocol.name,
        "train_fraction": float(train_fraction),
        "identity_fields": {
            "path": (
                "data/<format>/{raw,rgb}/ prefix dropped, variant_NN/lighting suffixes stripped"
            ),
            "metadata": list(IDENTITY_METADATA_KEYS),
        },
        "source_groups": ordered,
        "sides": sides,
        "counts": counts,
        "notes": notes,
    }


def write_split_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    """Write a split manifest as JSON so the exact split stays reproducible."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------------------
# Action/state normalisation (author statistics protocol)
# --------------------------------------------------------------------------------------


def _minmax(value: np.ndarray, low: Any, high: Any) -> np.ndarray:
    """Author min-max mapping into [-1, 1] with a 1e-6 span floor."""
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    scaled = 2.0 * (value - low_array) / np.maximum(high_array - low_array, 1.0e-6) - 1.0
    return np.clip(scaled, -1.0, 1.0)


def select_statistics(
    data: dict[str, Any],
    *,
    normalization: str,
    dataset_format: str,
    stats_key: str = "",
) -> dict[str, Any]:
    """Author ``_load_stats`` selection, including the dotted ``stats_key`` walk."""
    selected: Any = data
    if stats_key:
        for component in stats_key.split("."):
            if not isinstance(selected, dict) or component not in selected:
                raise KeyError(
                    f"stats_key={stats_key!r} component {component!r} is missing; available "
                    f"keys: {sorted(selected) if isinstance(selected, dict) else type(selected)}"
                )
            selected = selected[component]
        return selected
    if normalization == "starvla_minmax":
        protocol = resolve_protocol(dataset_format)
        key = "new_embodiment" if protocol.name == "robotwin2" else "franka"
        if key not in data:
            raise KeyError(
                f"starvla_minmax statistics expect top-level key {key!r}; available keys: "
                f"{sorted(data)}"
            )
        return data[key]
    if not isinstance(selected, dict):
        raise TypeError(f"statistics container must be a mapping, got {type(selected)}")
    return selected.get("norm_stats", selected)


@dataclass(frozen=True)
class ActionStateStatistics:
    """Selected author statistics plus how they were located."""

    normalization: str
    stats_path: Path | None
    action: dict[str, Any] | None
    state: dict[str, Any] | None
    action_stats_key: str
    state_stats_key: str
    source: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalization": self.normalization,
            "stats_path": None if self.stats_path is None else str(self.stats_path),
            "action_stats_key": self.action_stats_key,
            "state_stats_key": self.state_stats_key,
            "available": {
                "action": self.action is not None,
                "state": self.state is not None,
            },
            "fields": {
                "action": sorted(self.action) if isinstance(self.action, dict) else None,
                "state": sorted(self.state) if isinstance(self.state, dict) else None,
            },
            "source": dict(self.source),
        }


def load_statistics(
    stats_path: str | Path | None,
    *,
    normalization: str,
    dataset_format: str,
    stats_key: str = "",
    action_stats_key: str = "actions",
    state_stats_key: str = "state",
) -> ActionStateStatistics:
    """Read the author statistics file for the selected normalisation mode."""
    if normalization not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"unknown normalization {normalization!r}; expected one of "
            f"{list(SUPPORTED_NORMALIZATIONS)}"
        )
    path = None if stats_path in (None, "") else Path(stats_path)
    if normalization == "none":
        return ActionStateStatistics(
            normalization=normalization,
            stats_path=path,
            action=None,
            state=None,
            action_stats_key=action_stats_key,
            state_stats_key=state_stats_key,
            source={"mode": "none", "note": "raw actions/state are passed through unchanged"},
        )
    if path is None or not path.is_file():
        raise MissingRawVLAAssetError(
            rawvla_asset_requirements(
                dataset_format=dataset_format, stats_path=path, requires_stats=True
            ),
            dataset_format=dataset_format,
            root=default_rawvla_root(),
        )
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    selected = select_statistics(
        data, normalization=normalization, dataset_format=dataset_format, stats_key=stats_key
    )
    if not isinstance(selected, dict):
        raise TypeError(
            f"normalisation statistics at {path} resolved to {type(selected)}, expected a mapping"
        )
    action = selected.get(action_stats_key)
    state = selected.get(state_stats_key)
    if action is None:
        raise KeyError(
            f"action statistics key {action_stats_key!r} is missing from {path}; available "
            f"keys: {sorted(selected)}"
        )
    return ActionStateStatistics(
        normalization=normalization,
        stats_path=path,
        action=action,
        state=state,
        action_stats_key=action_stats_key,
        state_stats_key=state_stats_key,
        source={
            "stats_key": stats_key,
            "selection": "author _load_stats rules",
            "file": str(path),
        },
    )


def normalize_actions_and_state(
    actions: np.ndarray,
    state: np.ndarray | None,
    statistics: ActionStateStatistics,
    *,
    dataset_format: str,
    delta_action_indices: Sequence[int] = (),
    state_dim: int | None = None,
    joint_permutation: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Author action/state normalisation, with unsupported branches refused loudly.

    ``starvla_minmax`` for RoboTwin carries an AgileX training-order joint
    permutation in the author loader that cannot be verified from the local
    reference; it is refused unless the caller passes the explicit
    ``joint_permutation``, so no joint order is guessed.
    """
    mode = statistics.normalization
    protocol = resolve_protocol(dataset_format)
    actions = np.asarray(actions, dtype=np.float32).copy()
    raw_state = None if state is None else np.asarray(state, dtype=np.float32).copy()
    if mode == "none":
        return actions, raw_state
    if mode == "starvla_minmax":
        if protocol.name == "robotwin2" and joint_permutation is None:
            raise ValueError(
                "starvla_minmax for robotwin2 needs the author AgileX training-order joint "
                "permutation, which is not verifiable from the local reference; pass "
                "joint_permutation=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13) explicitly "
                "or use the verified zscore/quantile modes"
            )
        stats = statistics.action or {}
        if protocol.name == "robotwin2":
            order = np.asarray(joint_permutation, dtype=np.int64)
            actions = actions[:, order]
            dims = min(12, actions.shape[-1])
            actions[:, :dims] = _minmax(actions[:, :dims], stats["min"][:dims], stats["max"][:dims])
            actions[:, dims:] = (actions[:, dims:] > 0.49).astype(np.float32)
            return actions, raw_state
        dims = min(6, actions.shape[-1])
        actions[:, :dims] = _minmax(actions[:, :dims], stats["min"][:dims], stats["max"][:dims])
        if actions.shape[-1] > dims:
            actions[:, dims] = (actions[:, dims] + 1.0) * 0.5
        return actions, raw_state
    if mode in {"zscore", "pi0_zscore"}:
        indices = tuple(int(index) for index in delta_action_indices)
        if mode == "pi0_zscore" and not indices and raw_state is not None:
            indices = tuple(range(min(6, actions.shape[-1], raw_state.shape[-1])))
        if indices:
            if raw_state is None:
                raise ValueError("delta_action_indices need a current state to subtract")
            action_index = np.asarray(indices, dtype=np.int64)
            actions[:, action_index] -= raw_state[None, action_index]
        action_stats = statistics.action or {}
        if "mean" not in action_stats or "std" not in action_stats:
            raise KeyError(
                f"zscore normalisation needs mean/std in action statistics; available keys: "
                f"{sorted(action_stats)}"
            )
        action_dim = actions.shape[-1]
        actions = (actions - np.asarray(action_stats["mean"][:action_dim], dtype=np.float32)) / (
            np.maximum(np.asarray(action_stats["std"][:action_dim], dtype=np.float32), 1.0e-6)
        )
        if raw_state is None:
            return actions.astype(np.float32), None
        state_stats = statistics.state or {}
        if "mean" not in state_stats or "std" not in state_stats:
            raise KeyError(
                f"zscore normalisation needs mean/std in state statistics; available keys: "
                f"{sorted(state_stats)}"
            )
        dim = raw_state.shape[-1] if state_dim is None else int(state_dim)
        normalized_state = (
            raw_state - np.asarray(state_stats["mean"][:dim], dtype=np.float32)
        ) / np.maximum(np.asarray(state_stats["std"][:dim], dtype=np.float32), 1.0e-6)
        return actions.astype(np.float32), normalized_state.astype(np.float32)
    if mode in {"quantile", "pi05_quantile"}:
        action_stats = statistics.action or {}
        if "q01" not in action_stats or "q99" not in action_stats:
            raise KeyError(
                f"quantile normalisation needs q01/q99 in action statistics; available keys: "
                f"{sorted(action_stats)}"
            )
        action_dim = actions.shape[-1]
        actions = _minmax(
            actions,
            action_stats["q01"][:action_dim],
            action_stats["q99"][:action_dim],
        ).astype(np.float32)
        if raw_state is None:
            return actions, None
        state_stats = statistics.state or {}
        if "q01" not in state_stats or "q99" not in state_stats:
            raise KeyError(
                f"quantile normalisation needs q01/q99 in state statistics; available keys: "
                f"{sorted(state_stats)}"
            )
        dim = raw_state.shape[-1] if state_dim is None else int(state_dim)
        normalized_state = _minmax(
            raw_state, state_stats["q01"][:dim], state_stats["q99"][:dim]
        ).astype(np.float32)
        return actions, normalized_state
    raise ValueError(f"unsupported normalization {mode!r}")


# --------------------------------------------------------------------------------------
# Measurement windows
# --------------------------------------------------------------------------------------


@dataclass
class MeasurementWindow:
    """One camera's causal measurement window; field semantics follow ``MeasurementFrame``.

    ``values``/``valid_mask``/``bad_mask``/``saturated_mask`` carry a leading time
    axis ``[T, C, H, W]`` (one frame at time ``t`` is ``[t:t+1]``).  Every frame
    keeps ``episode_ids``/``frame_ids``; repeated leading padding frames are marked
    ``new_measurement=False`` so they never count as new evidence twice.  Action
    supervision lives on the window as well, with an independent ``action_mask``
    for padded action tails.  ``sample_xy`` records the per-channel
    ``(row, col)`` sampling offset: zeros for co-located three-channel pseudo-RAW
    and the real CFA plane phase when the window was built from a CFA plane stack.
    Extra OWAC fields (``variance``, actions, state) are extensions of the frozen
    contract, not replacements for it; use :meth:`to_measurement_frame` to hand a
    single time step to :class:`~owac.core.encoder_contract.MeasurementFrame`.
    """

    values: torch.Tensor
    valid_mask: torch.Tensor
    bad_mask: torch.Tensor
    saturated_mask: torch.Tensor
    episode_ids: torch.Tensor
    frame_ids: torch.Tensor
    camera_id: str = ""
    timestamp: torch.Tensor | None = None
    time_unit: str = "frame"
    new_measurement: torch.Tensor | None = None
    missing_mask: torch.Tensor | None = None
    variance: torch.Tensor | None = None
    sample_xy: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    action_indices: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_available: bool = False
    state_kind: str = ""
    executed_action: torch.Tensor | None = None
    executed_action_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_available: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 4:
            raise ValueError(
                "MeasurementWindow.values must be [T, C, H, W]; got shape "
                f"{tuple(self.values.shape)}"
            )
        frames = int(self.values.shape[0])
        for name in ("valid_mask", "bad_mask", "saturated_mask", "missing_mask", "variance"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            if tuple(tensor.shape) != tuple(self.values.shape):
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} does not match values "
                    f"{tuple(self.values.shape)}"
                )
        for name in ("episode_ids", "frame_ids", "new_measurement", "timestamp"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            if int(tensor.shape[0]) != frames:
                raise ValueError(
                    f"{name} must have one entry per frame ({frames}); got {tuple(tensor.shape)}"
                )

    @property
    def frame_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def channels(self) -> int:
        return int(self.values.shape[1])

    @property
    def camera_ids(self) -> list[str]:
        return [self.camera_id]

    def unique_frame_count(self) -> int:
        """Number of distinct frames the window really measures."""
        if self.new_measurement is None:
            return len(set(int(value) for value in self.frame_ids.tolist()))
        return int(self.new_measurement.sum().item())

    def measurement_weights(self) -> torch.Tensor:
        """``[T]`` FP32 weight that counts each unique frame exactly once."""
        if self.new_measurement is None:
            return torch.ones(self.frame_count, dtype=torch.float32)
        return self.new_measurement.to(torch.float32)

    def to_measurement_frame(self, time_index: int = 0) -> MeasurementFrame:
        """One time step as the frozen contract's ``MeasurementFrame`` (batch size 1)."""
        index = int(time_index)
        if not 0 <= index < self.frame_count:
            raise IndexError(f"time_index {index} outside the window of {self.frame_count} frames")
        step = slice(index, index + 1)
        return MeasurementFrame(
            values=self.values[step],
            valid_mask=self.valid_mask[step],
            bad_mask=self.bad_mask[step],
            saturated_mask=self.saturated_mask[step],
            sample_xy=self.sample_xy,
            episode_ids=self.episode_ids[step] if self.episode_ids is not None else None,
            frame_ids=self.frame_ids[step] if self.frame_ids is not None else None,
            camera_id=self.camera_id,
            timestamp=self.timestamp[step] if self.timestamp is not None else None,
            time_unit=self.time_unit,
            new_measurement=(
                self.new_measurement[step] if self.new_measurement is not None else None
            ),
            metadata=dict(self.metadata),
            metadata_available=dict(self.metadata_available),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "values_shape": list(self.values.shape),
            "frame_ids": self.frame_ids.tolist(),
            "episode_ids": self.episode_ids.tolist(),
            "frame_count": self.frame_count,
            "unique_frames": self.unique_frame_count(),
            "time_unit": self.time_unit,
            "new_measurement": (
                None if self.new_measurement is None else self.new_measurement.tolist()
            ),
            "action_mask": None if self.action_mask is None else self.action_mask.tolist(),
            "metadata": dict(self.metadata),
            "metadata_available": dict(self.metadata_available),
        }


@dataclass
class MeasurementWindowBatch:
    """A batch of windows: every field gains a leading batch axis ``[B, ...]``."""

    values: torch.Tensor
    valid_mask: torch.Tensor
    bad_mask: torch.Tensor
    saturated_mask: torch.Tensor
    episode_ids: torch.Tensor
    frame_ids: torch.Tensor
    new_measurement: torch.Tensor
    camera_id: str
    time_unit: str
    timestamp: torch.Tensor | None = None
    missing_mask: torch.Tensor | None = None
    variance: torch.Tensor | None = None
    sample_xy: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    action_indices: torch.Tensor | None = None
    metadata: list[dict[str, Any]] = field(default_factory=list)


def collate_measurement_windows(windows: Sequence[MeasurementWindow]) -> MeasurementWindowBatch:
    """Stack windows from one camera into a ``[B, T, C, H, W]`` batch."""

    def stack(name: str, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        tensors = [getattr(window, name) for window in windows]
        if any(tensor is None for tensor in tensors):
            return None
        stacked = torch.stack([tensor for tensor in tensors if tensor is not None])
        return stacked if dtype is None else stacked.to(dtype)

    if not windows:
        raise ValueError("cannot collate an empty window list")
    cameras = {window.camera_id for window in windows}
    if len(cameras) != 1:
        raise ValueError(f"one batch holds one camera; got {sorted(cameras)}")
    time_units = {window.time_unit for window in windows}
    if len(time_units) != 1:
        raise ValueError(f"one batch holds one time unit; got {sorted(time_units)}")
    first = windows[0]
    sample_xy = first.sample_xy
    if sample_xy is not None:
        for window in windows[1:]:
            if window.sample_xy is None or not bool(torch.equal(window.sample_xy, sample_xy)):
                raise ValueError("sampling offsets differ inside one batch")
    values = stack("values")
    valid_mask = stack("valid_mask")
    bad_mask = stack("bad_mask")
    saturated_mask = stack("saturated_mask")
    if values is None or valid_mask is None or bad_mask is None or saturated_mask is None:
        raise ValueError("values, valid_mask, bad_mask and saturated_mask are required")
    return MeasurementWindowBatch(
        values=values,
        valid_mask=valid_mask,
        bad_mask=bad_mask,
        saturated_mask=saturated_mask,
        episode_ids=stack("episode_ids"),
        frame_ids=stack("frame_ids"),
        new_measurement=stack("new_measurement"),
        camera_id=first.camera_id,
        time_unit=first.time_unit,
        timestamp=stack("timestamp"),
        missing_mask=stack("missing_mask"),
        variance=stack("variance"),
        sample_xy=sample_xy,
        action=stack("action"),
        action_mask=stack("action_mask"),
        action_indices=stack("action_indices"),
        metadata=[dict(window.metadata) for window in windows],
    )


def measurement_window_from_cfa(
    planes: CfaPlaneStack,
    *,
    camera_id: str = "cfa",
    episode_id: int = 0,
    frame_id: int = 0,
    timestamp: float | None = None,
    time_unit: str = "frame",
    new_measurement: bool = True,
) -> MeasurementWindow:
    """Wrap four CFA measurement planes as a one-frame window.

    The four planes keep their own ``sample_xy`` and colour names, so downstream
    code samples them separately and mixes features afterwards; they are never
    four co-located colours of one pixel.
    """
    if time_unit not in {"frame", "second"}:
        raise ValueError(f"time_unit must be 'frame' or 'second', got {time_unit!r}")
    if time_unit == "second" and timestamp is None:
        raise ValueError("time_unit='second' requires a timestamp; do not invent seconds")
    stamp = float(frame_id) if timestamp is None else float(timestamp)
    metadata = dict(planes.metadata)
    metadata["value_units"] = planes.metadata.get("calibration", "x=(raw-black)/(white-black)")
    metadata_available = {
        "timestamp": timestamp is not None,
        "timestamp_source": "declared" if timestamp is not None else "frame_index",
        "saturation_level": bool(planes.metadata.get("saturation_level_available", False)),
        "bad_pixel_mask": bool(planes.bad.any().item()),
        "executed_action": False,
        "co_located_channels": False,
    }
    return MeasurementWindow(
        values=planes.values.unsqueeze(0),
        valid_mask=planes.valid.unsqueeze(0),
        bad_mask=planes.bad.unsqueeze(0),
        saturated_mask=planes.saturated.unsqueeze(0),
        missing_mask=planes.missing.unsqueeze(0),
        sample_xy=planes.sample_xy,
        episode_ids=torch.tensor([int(episode_id)], dtype=torch.int64),
        frame_ids=torch.tensor([int(frame_id)], dtype=torch.int64),
        camera_id=camera_id,
        timestamp=torch.tensor([stamp], dtype=torch.float32),
        time_unit=time_unit,
        new_measurement=torch.tensor([bool(new_measurement)], dtype=torch.bool),
        metadata=metadata,
        metadata_available=metadata_available,
    )


# --------------------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------------------


@dataclass
class SequenceSample:
    """One training sample: per-camera windows plus action/state supervision.

    The action and state tensors are shared by reference with every camera window
    of this sample, so a window stays self-contained without copying the action
    block per camera.  Do not mutate them in place.
    """

    windows: dict[str, MeasurementWindow]
    action: torch.Tensor
    action_raw: torch.Tensor
    action_mask: torch.Tensor
    action_indices: torch.Tensor
    state: torch.Tensor | None
    state_raw: torch.Tensor | None
    state_available: bool
    state_kind: str
    proprio_available: bool
    executed_action: torch.Tensor | None
    executed_action_available: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def camera_ids(self) -> list[str]:
        return list(self.windows)

    def window(self, camera_id: str) -> MeasurementWindow:
        if camera_id not in self.windows:
            raise KeyError(
                f"camera {camera_id!r} is not in this sample; available: {sorted(self.windows)}"
            )
        return self.windows[camera_id]

    def as_embodiment_context(self) -> EmbodimentContext:
        """Body context with explicit availability flags and no fabricated actions."""
        return EmbodimentContext(
            proprio=self.state,
            proprio_available=torch.tensor([bool(self.state_available)], dtype=torch.bool),
            executed_action=self.executed_action,
            executed_action_available=torch.tensor(
                [bool(self.executed_action_available)], dtype=torch.bool
            ),
            camera_transform=None,
            camera_transform_available=torch.tensor([False], dtype=torch.bool),
            task_tokens=None,
            task_mask=None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cameras": sorted(self.windows),
            "action_shape": list(self.action.shape),
            "action_mask": self.action_mask.tolist(),
            "action_indices": self.action_indices.tolist(),
            "state_available": bool(self.state_available),
            "state_kind": self.state_kind,
            "proprio_available": bool(self.proprio_available),
            "executed_action_available": bool(self.executed_action_available),
            "meta": dict(self.meta),
        }


def collate_sequence_samples(samples: Sequence[SequenceSample]) -> dict[str, Any]:
    """Collate samples for a DataLoader: one ``[B, ...]`` batch per camera."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    cameras = sorted(samples[0].windows)
    for sample in samples[1:]:
        if sorted(sample.windows) != cameras:
            raise ValueError(
                "every sample in a batch must hold the same cameras; got "
                f"{cameras} and {sorted(sample.windows)}"
            )
    windows = {
        camera: collate_measurement_windows([sample.windows[camera] for sample in samples])
        for camera in cameras
    }
    executed = [sample.executed_action for sample in samples]
    return {
        "windows": windows,
        "action": torch.stack([sample.action for sample in samples]),
        "action_raw": torch.stack([sample.action_raw for sample in samples]),
        "action_mask": torch.stack([sample.action_mask for sample in samples]),
        "action_indices": torch.stack([sample.action_indices for sample in samples]),
        "state": (
            None
            if any(value is None for value in [sample.state for sample in samples])
            else torch.stack([sample.state for sample in samples if sample.state is not None])
        ),
        "executed_action": (
            None if any(value is None for value in executed) else torch.stack(executed)
        ),
        "executed_action_available": torch.tensor(
            [bool(sample.executed_action_available) for sample in samples], dtype=torch.bool
        ),
        "meta": [dict(sample.meta) for sample in samples],
    }


@dataclass(frozen=True)
class WindowPlan:
    """Where one rollout window sits, and whether state may be carried into it."""

    episode_index: int
    episode_key: str
    current_frame: int
    window_frames: tuple[int, ...]
    continues_within_episode: bool
    reset_required: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "episode_key": self.episode_key,
            "current_frame": self.current_frame,
            "window_frames": list(self.window_frames),
            "continues_within_episode": bool(self.continues_within_episode),
            "reset_required": bool(self.reset_required),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class RawVLASequenceDataset(Dataset):
    """Causal RawVLA NPZ windows that emit measurement tensors, never images.

    Parameters mirror the author loader where the data protocol is reused:
    ``observation_stride``/``action_horizon``/``action_offset``/``state_source_key``
    and the normalisation statistics.  Sampling is deterministic per index
    (``seed + index*104729``), the split filters the episode list by source group,
    and every window starts with ``new_measurement[0] is True`` so a random batch
    can reset its recurrent state at the window start while a continuous rollout
    only carries state inside one episode.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        dataset_format: str = "libero",
        split: str = "train",
        window: int = 8,
        observation_stride: int | None = None,
        action_horizon: int | None = None,
        action_offset: int | None = None,
        state_source_key: str | None = None,
        normalization: str | None = None,
        stats_path: str | Path | None = None,
        stats_key: str = "",
        action_stats_key: str = "actions",
        state_stats_key: str = "state",
        delta_action_indices: Sequence[int] = (),
        joint_permutation: Sequence[int] | None = None,
        camera_keys: Sequence[str] | None = None,
        raw_cache_root: str | Path | None = None,
        rgb_cache_root: str | Path | None = None,
        saturated_level: float | None = None,
        noise_model: NoiseModel | None = None,
        declared_calibration: bool = True,
        split_version: str = "n01-rawvla-split-v1",
        train_fraction: float = 0.8,
        split_manifest: dict[str, Any] | None = None,
        split_manifest_path: str | Path | None = None,
        seed: int = 42,
        samples_per_episode: int = 32,
        virtual_length: int | None = None,
        fixture_provenance: str = "auto",
        range_evidence_path: str | Path | None = None,
    ) -> None:
        protocol = resolve_protocol(dataset_format)
        self.protocol = protocol
        self.root = Path(root) if root is not None else default_rawvla_root()
        self.raw_cache_root = (
            Path(raw_cache_root)
            if raw_cache_root is not None
            else default_raw_cache_root(self.root, protocol.name)
        )
        self.rgb_cache_root = (
            Path(rgb_cache_root)
            if rgb_cache_root is not None
            else default_rgb_cache_root(self.root, protocol.name)
        )
        self.normalization = protocol.normalization if normalization is None else str(normalization)
        self.stats_path = (
            Path(stats_path)
            if stats_path is not None
            else default_stats_path(self.root, protocol.name)
        )
        self.stats_key = stats_key
        self.action_stats_key = action_stats_key
        self.state_stats_key = state_stats_key
        self.delta_action_indices = tuple(int(index) for index in delta_action_indices)
        self.joint_permutation = (
            None if joint_permutation is None else tuple(int(i) for i in joint_permutation)
        )
        self.camera_keys = tuple(camera_keys) if camera_keys else protocol.camera_keys
        self.state_source_key = state_source_key or protocol.state_source_key
        self.window = int(window)
        self.observation_stride = (
            protocol.observation_stride if observation_stride is None else int(observation_stride)
        )
        self.action_horizon = (
            protocol.action_horizon if action_horizon is None else int(action_horizon)
        )
        self.action_offset = protocol.action_offset if action_offset is None else int(action_offset)
        self.saturated_level = saturated_level
        self.noise_model = noise_model
        self.declared_calibration = bool(declared_calibration)
        self.seed = int(seed)
        self.samples_per_episode = int(samples_per_episode)
        self.split_version = split_version
        self.train_fraction = float(train_fraction)
        self.split_manifest_path = (
            None if split_manifest_path is None else Path(split_manifest_path)
        )
        self.fixture_provenance = str(fixture_provenance)
        self.range_evidence_path = (
            None if range_evidence_path is None else Path(range_evidence_path)
        )
        self._episode_cache: dict[str, dict[str, np.ndarray]] = {}
        self._cached_path: str | None = None
        self._fixture_paths: set[str] = set()
        self._timestamp_paths: set[str] = set()
        self._provenance_checked: set[str] = set()
        self._time_units: dict[str, int] = {}
        self._scanned_metadata: dict[str, dict[str, Any]] = {}
        self.split_name = "unknown"

        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if self.observation_stride < 1:
            raise ValueError(f"observation_stride must be >= 1, got {observation_stride}")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        if self.action_offset < 0:
            raise ValueError(f"action_offset must be >= 0, got {action_offset}")
        if self.samples_per_episode < 1:
            raise ValueError(f"samples_per_episode must be >= 1, got {samples_per_episode}")
        if self.fixture_provenance not in {"auto", "synthetic", "real"}:
            raise ValueError(
                "fixture_provenance must be 'auto', 'synthetic' or 'real'; got "
                f"{fixture_provenance!r}"
            )
        if protocol.camera_dtype == "uint8" and self.normalization == "none":
            raise ValueError(
                "normalization='none' cannot be used for the LIBERO uint8 pseudo-RAW cache: "
                "uint8 digital numbers must be divided by 255 before they are measurements"
            )

        self.episodes = sorted(
            str(path) for path in self.raw_cache_root.glob("**/*.npz") if path.is_file()
        )
        if not self.episodes:
            raise MissingRawVLAAssetError(
                rawvla_asset_requirements(
                    root=self.root,
                    dataset_format=protocol.name,
                    raw_cache_root=self.raw_cache_root,
                    stats_path=self.stats_path,
                    rgb_cache_root=self.rgb_cache_root,
                    requires_stats=self.normalization != "none",
                ),
                dataset_format=protocol.name,
                root=self.root,
            )
        if self.normalization != "none" and not self.stats_path.is_file():
            raise MissingRawVLAAssetError(
                rawvla_asset_requirements(
                    root=self.root,
                    dataset_format=protocol.name,
                    raw_cache_root=self.raw_cache_root,
                    stats_path=self.stats_path,
                    rgb_cache_root=self.rgb_cache_root,
                    requires_stats=True,
                ),
                dataset_format=protocol.name,
                root=self.root,
            )
        self.statistics = load_statistics(
            self.stats_path,
            normalization=self.normalization,
            dataset_format=protocol.name,
            stats_key=stats_key,
            action_stats_key=action_stats_key,
            state_stats_key=state_stats_key,
        )
        if split_manifest is not None:
            self.split = dict(split_manifest)
        elif self.split_manifest_path is not None and self.split_manifest_path.is_file():
            self.split = load_split_manifest(self.split_manifest_path)
        else:
            self.split = build_split_manifest(
                self.episodes,
                dataset_format=protocol.name,
                relative_to=self.raw_cache_root,
                train_fraction=self.train_fraction,
                split_version=split_version,
            )
        self._episode_index = {path: index for index, path in enumerate(self.episodes)}
        self._split_of_path: dict[str, str] = {}
        for side, files in self.split.get("sides", {}).items():
            for entry in files:
                self._split_of_path[str(entry)] = side
        self.paths = self._paths_for_split(split)
        self.virtual_length = (
            int(virtual_length)
            if virtual_length is not None
            else len(self.paths) * self.samples_per_episode
        )
        if self.virtual_length < 1:
            raise ValueError(f"virtual_length must be >= 1, got {virtual_length}")

    # -- splits ------------------------------------------------------------------

    def _paths_for_split(self, split: str) -> list[str]:
        key = str(split).lower()
        if key in {"all", "*"}:
            self.split_name = "all"
            return list(self.episodes)
        if key in {"val", "valid", "validation", "dev"}:
            key = "validation"
        sides = self.split.get("sides", {})
        if key not in sides:
            raise ValueError(
                f"unknown split {split!r}; expected 'train', 'validation' or 'all' "
                f"(manifest sides: {sorted(sides)})"
            )
        selected = sorted(str(path) for path in sides[key])
        self.split_name = key
        if not selected:
            raise ValueError(
                f"split {key!r} is empty for split_version={self.split.get('split_version')!r}; "
                f"counts={self.split.get('counts')}"
            )
        return selected

    def write_split_manifest(self, path: str | Path | None = None) -> Path:
        """Write the reproducible split inventory; returns the JSON path."""
        target = (
            Path(path)
            if path is not None
            else (self.raw_cache_root.parent / "splits" / f"{self.split_version}.json")
        )
        return write_split_manifest(target, self.split)

    def plan_rollout(
        self,
        episode_index: int,
        *,
        start_frame: int = 0,
        windows: int = 4,
        stride: int | None = None,
    ) -> list[WindowPlan]:
        """Causal window plan inside one episode; never steps across episodes.

        The first window always requires a state reset and later windows carry
        state only because they overlap the previous one inside the same episode.
        A trainer can use the first ``burn_in`` windows to build state and only
        compute the loss on the frames after them.
        """
        episodes = self.episodes
        index = int(episode_index)
        if not 0 <= index < len(episodes):
            raise IndexError(f"episode_index {index} outside 0..{len(episodes) - 1}")
        path = episodes[index]
        arrays = self._load_episode(path)
        num_frames = int(arrays["actions"].shape[0])
        step = self.observation_stride if stride is None else int(stride)
        if step < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        plans: list[WindowPlan] = []
        previous_last: int | None = None
        for window_index in range(int(windows)):
            current = int(start_frame) + window_index * step
            if current >= num_frames:
                break
            frames = tuple(int(value) for value in self._window_indices(current, num_frames))
            continues = (
                previous_last is not None
                and window_index > 0
                and frames
                and frames[0] <= previous_last
            )
            plans.append(
                WindowPlan(
                    episode_index=index,
                    episode_key=self._episode_key(path),
                    current_frame=current,
                    window_frames=frames,
                    continues_within_episode=bool(continues),
                    reset_required=not bool(continues),
                    reason=(
                        "same episode and the new window overlaps the previous one, so the "
                        "recurrent state may be carried"
                        if continues
                        else "window start (or first window of this episode): reset the "
                        "recurrent state"
                    ),
                )
            )
            previous_last = frames[-1]
        return plans

    # -- sampling ----------------------------------------------------------------

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, index: int) -> SequenceSample:
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an integer, got {type(index).__name__}")
        if index < 0 or index >= self.virtual_length:
            raise IndexError(f"index {index} outside dataset length {self.virtual_length}")
        episode_slot = (int(index) // self.samples_per_episode) % len(self.paths)
        rng = np.random.default_rng(self.seed + int(index) * 104729)
        path = self.paths[episode_slot]
        arrays = self._load_episode(path)
        num_frames = int(arrays["actions"].shape[0])
        current = int(rng.integers(0, num_frames))
        return self.build_sample(path, arrays, current)

    def sample_at(self, episode_index: int, current_frame: int) -> SequenceSample:
        """Deterministic window at an explicit episode/frame, for tests and rollouts.

        ``episode_index`` indexes :attr:`episodes`, the full inventory, while
        ``__getitem__`` samples inside the requested split; the window itself is
        always causal and never crosses the episode boundary.
        """
        index = int(episode_index)
        if not 0 <= index < len(self.episodes):
            raise IndexError(f"episode_index {index} outside 0..{len(self.episodes) - 1}")
        path = self.episodes[index]
        arrays = self._load_episode(path)
        num_frames = int(arrays["actions"].shape[0])
        frame = int(current_frame)
        if not 0 <= frame < num_frames:
            raise IndexError(
                f"current_frame {frame} outside episode {path} with {num_frames} frames"
            )
        return self.build_sample(path, arrays, frame)

    def build_sample(
        self,
        path: str,
        arrays: dict[str, np.ndarray],
        current_frame: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SequenceSample:
        """Assemble one window from an already loaded episode dictionary."""
        num_frames = int(arrays["actions"].shape[0])
        metadata = self._episode_metadata(path, arrays) if metadata is None else metadata
        episode_key = self._episode_key(path)
        episode_id = self._episode_index[path]
        frame_indices = self._window_indices(current_frame, num_frames)
        frame_ids = torch.tensor(frame_indices, dtype=torch.int64)
        new_measurement = torch.tensor(
            [True]
            + [
                bool(frame_indices[k] != frame_indices[k - 1]) for k in range(1, len(frame_indices))
            ],
            dtype=torch.bool,
        )
        leading_offsets = self.observation_stride * np.arange(self.window - 1, -1, -1)
        padding_frames = int(np.sum(current_frame - leading_offsets < 0))

        action_indices = current_frame + self.action_offset + np.arange(self.action_horizon)
        action_in_range = (action_indices >= 0) & (action_indices < num_frames)
        clipped_actions = np.clip(action_indices, 0, num_frames - 1)

        if "actions" not in arrays:
            raise KeyError(
                f"episode {path} has no 'actions' array; available keys: {sorted(arrays)}"
            )
        action_raw = np.asarray(arrays["actions"][clipped_actions], dtype=np.float32)
        state_raw, state_kind = self._read_state(path, arrays, current_frame)

        timestamps, time_unit, timestamp_available = self._read_timestamps(
            path, arrays, frame_indices, num_frames
        )
        calibration = self._declared_calibration(metadata) if self.declared_calibration else None

        normalized_action, normalized_state = normalize_actions_and_state(
            action_raw,
            state_raw,
            self.statistics,
            dataset_format=self.protocol.name,
            delta_action_indices=self.delta_action_indices,
            state_dim=self.protocol.state_dim,
            joint_permutation=self.joint_permutation,
        )
        action = torch.from_numpy(np.ascontiguousarray(normalized_action, dtype=np.float32))
        action_raw_tensor = torch.from_numpy(np.ascontiguousarray(action_raw, dtype=np.float32))
        action_mask = torch.from_numpy(np.ascontiguousarray(action_in_range, dtype=bool))
        state_tensor = (
            None
            if normalized_state is None
            else torch.from_numpy(np.ascontiguousarray(normalized_state, dtype=np.float32))
        )
        state_raw_tensor = (
            None if state_raw is None else torch.from_numpy(np.ascontiguousarray(state_raw))
        )

        windows: dict[str, MeasurementWindow] = {}
        for camera_key in self.camera_keys:
            if camera_key not in arrays:
                raise KeyError(
                    f"camera key {camera_key!r} is missing from {path}; available keys: "
                    f"{sorted(arrays)}"
                )
            stored = np.asarray(arrays[camera_key])
            values, saturated = self._camera_values(stored, camera_key, calibration)
            if (
                self.protocol.camera_dtype != "uint8"
                and calibration is None
                and metadata.get("representation") == "three_channel_linear_float32"
                and metadata.get("raw_white_level") is not None
            ):
                saturated = stored >= float(metadata["raw_white_level"])
            values = np.ascontiguousarray(values[frame_indices].transpose(0, 3, 1, 2))
            saturated = np.ascontiguousarray(saturated[frame_indices].transpose(0, 3, 1, 2))
            values_tensor = torch.from_numpy(values)
            saturated_tensor = torch.from_numpy(saturated.astype(bool))
            shape = tuple(values.shape)
            camera_id = self.protocol.camera_id(camera_key)
            variance = (
                None if self.noise_model is None else self.noise_model.variance(values_tensor)
            )
            window_metadata = dict(metadata)
            window_metadata.update(
                {
                    "source_path": path,
                    "camera_key": camera_key,
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "split": self._split_of(path),
                    "value_units": self.protocol.raw_unit,
                    "digital_number_dtype": str(stored.dtype),
                    "raw_scaling": self.protocol.raw_scaling,
                    "raw_range_evidence": self.protocol.raw_range_evidence,
                    "sampling_geometry": "per_pixel_trichromatic_collocated",
                    "co_located_channels": True,
                    "current_frame": int(current_frame),
                    "window_frames": [int(value) for value in frame_indices],
                    "num_frames": num_frames,
                    "padded_leading_frames": padding_frames,
                    "action_window": [int(clipped_actions[0]), int(clipped_actions[-1])],
                    "action_in_range_steps": int(action_in_range.sum()),
                    "action_indices_semantics": (
                        "absolute frame indices clipped to the episode end; steps with "
                        "action_mask False are padding and are never supervised"
                    ),
                    "action_offset": self.action_offset,
                    "action_horizon": self.action_horizon,
                    "normalization": self.normalization,
                    "declared_calibration": calibration,
                    "synthetic_fixture": bool(metadata.get("synthetic_fixture", False)),
                    "reset_required_at_window_start": True,
                }
            )
            window_metadata_available = self._metadata_availability(
                metadata, timestamp_available, calibration
            )
            windows[camera_id] = MeasurementWindow(
                values=values_tensor,
                valid_mask=torch.ones(shape, dtype=torch.bool),
                bad_mask=torch.zeros(shape, dtype=torch.bool),
                saturated_mask=saturated_tensor,
                missing_mask=torch.zeros(shape, dtype=torch.bool),
                sample_xy=torch.zeros((shape[1], 2), dtype=torch.int64),
                episode_ids=torch.full((len(frame_indices),), episode_id, dtype=torch.int64),
                frame_ids=frame_ids.clone(),
                camera_id=camera_id,
                timestamp=timestamps.clone(),
                time_unit=time_unit,
                new_measurement=new_measurement.clone(),
                variance=variance,
                action=action,
                action_mask=action_mask,
                action_indices=torch.from_numpy(
                    np.ascontiguousarray(clipped_actions, dtype=np.int64)
                ),
                state=state_tensor,
                state_available=state_tensor is not None,
                state_kind=state_kind,
                executed_action=None,
                executed_action_available=False,
                metadata=window_metadata,
                metadata_available=window_metadata_available,
            )

        self._time_units[time_unit] = self._time_units.get(time_unit, 0) + 1
        meta = {
            "episode_path": path,
            "episode_key": episode_key,
            "episode_id": episode_id,
            "split": self._split_of(path),
            "current_frame": int(current_frame),
            "num_frames": num_frames,
            "window_frames": [int(value) for value in frame_indices],
            "new_measurement": new_measurement.tolist(),
            "padded_leading_frames": padding_frames,
            "action_indices": [int(value) for value in clipped_actions],
            "action_in_range": action_in_range.tolist(),
            "action_offset": self.action_offset,
            "action_horizon": self.action_horizon,
            "observation_stride": self.observation_stride,
            "normalization": self.normalization,
            "state_kind": state_kind,
            "executed_action_available": False,
            "executed_action_note": (
                "the author cache stores commands and future supervision, not a measured "
                "executed action, so executed_action stays None and the flag stays False"
            ),
            "task_description": str(
                metadata.get("task_description") or metadata.get("task_name") or ""
            ),
            "lighting_domain": metadata.get("lighting_domain"),
            "synthetic_fixture": bool(metadata.get("synthetic_fixture", False)),
            "fixture_note": metadata.get("fixture_note"),
            "noise_model": None if self.noise_model is None else self.noise_model.as_dict(),
            "time_unit": time_unit,
            "timestamp_available": timestamp_available,
        }
        return SequenceSample(
            windows=windows,
            action=action,
            action_raw=action_raw_tensor,
            action_mask=action_mask,
            action_indices=torch.from_numpy(np.ascontiguousarray(clipped_actions, dtype=np.int64)),
            state=state_tensor,
            state_raw=state_raw_tensor,
            state_available=state_tensor is not None,
            state_kind=state_kind,
            proprio_available=self.protocol.proprio_available,
            executed_action=None,
            executed_action_available=False,
            meta=meta,
        )

    # -- helpers -----------------------------------------------------------------

    def _window_indices(self, current_frame: int, num_frames: int) -> np.ndarray:
        offsets = self.observation_stride * np.arange(self.window - 1, -1, -1)
        return np.clip(int(current_frame) - offsets, 0, num_frames - 1)

    def _camera_values(
        self,
        stored: np.ndarray,
        camera_key: str,
        calibration: dict[str, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if stored.ndim != 4 or stored.shape[-1] != 3:
            raise ValueError(
                f"camera key {camera_key!r} must be [T, H, W, 3]; got shape {tuple(stored.shape)}"
            )
        if self.protocol.camera_dtype == "uint8":
            if stored.dtype != np.uint8:
                raise ValueError(
                    f"camera key {camera_key!r} is stored as {stored.dtype}, but the pinned "
                    "LIBERO pseudo-RAW chain stores uint8 and its 0..255 view is the only "
                    "verified scale; refusing to guess a scale for another dtype"
                )
            values = stored.astype(np.float32) / 255.0
            level = 255.0 if self.saturated_level is None else float(self.saturated_level)
            saturated = np.asarray(stored) >= level
            return values, saturated
        if stored.dtype.kind != "f":
            raise ValueError(
                f"camera key {camera_key!r} is stored as {stored.dtype}, but the RoboTwin cache "
                "stores HDR-derived float32 RAW; refusing to reinterpret another dtype"
            )
        values = stored.astype(np.float32)
        if calibration is not None:
            black = float(calibration["black"])
            white = float(calibration["white"])
            values = (values - black) / (white - black)
            saturated = stored.astype(np.float32) >= white
            return values, saturated
        if self.saturated_level is None:
            return values, np.zeros(values.shape, dtype=bool)
        return values, stored.astype(np.float32) >= float(self.saturated_level)

    def _read_state(
        self,
        path: str,
        arrays: dict[str, np.ndarray],
        current_frame: int,
    ) -> tuple[np.ndarray | None, str]:
        key = self.state_source_key
        if key not in arrays:
            raise KeyError(
                f"state_source_key={key!r} is missing from {path}; available keys: {sorted(arrays)}"
            )
        stored = np.asarray(arrays[key])
        if stored.ndim != 2:
            raise ValueError(
                f"state_source_key={key!r} must be [T, S]; got shape {tuple(stored.shape)}"
            )
        dim = self.protocol.state_dim
        if stored.shape[-1] < dim:
            raise ValueError(
                f"state_source_key={key!r} stores {stored.shape[-1]} dims but the protocol "
                f"needs {dim}"
            )
        state = stored[current_frame, :dim].astype(np.float32)
        kind = self.protocol.state_kind
        if key != self.protocol.state_source_key:
            kind = f"cached_channel:{key}"
        return state, kind

    def _read_timestamps(
        self,
        path: str,
        arrays: dict[str, np.ndarray],
        frame_indices: np.ndarray,
        num_frames: int,
    ) -> tuple[torch.Tensor, str, bool]:
        for key in TIMESTAMP_KEYS:
            if key not in arrays:
                continue
            stored = np.asarray(arrays[key])
            if stored.ndim != 1 or stored.shape[0] != num_frames:
                continue
            if stored.dtype.kind not in "fiu":
                continue
            values = stored.astype(np.float64)
            if not np.all(np.isfinite(values)):
                continue
            if num_frames > 1 and np.any(np.diff(values) < 0):
                continue
            self._timestamp_paths.add(path)
            return (
                torch.tensor(values[frame_indices], dtype=torch.float32),
                "second",
                True,
            )
        return torch.tensor(frame_indices, dtype=torch.float32), "frame", False

    def _declared_calibration(self, metadata: dict[str, Any]) -> dict[str, float] | None:
        """Only an explicitly declared black/white pair may rescale float32 RAW."""
        black = metadata.get("raw_black_level")
        white = metadata.get("raw_white_level")
        # The released RoboTwin linear-float cache declares a clipping ceiling,
        # not a black/white sensor calibration. Preserve its author numeric unit.
        if black is None and metadata.get("representation") == "three_channel_linear_float32":
            return None
        if black is None and white is None:
            return None
        if black is None or white is None:
            raise ValueError(
                "declared RAW calibration is incomplete: both raw_black_level and "
                f"raw_white_level are needed, got black={black!r} white={white!r}"
            )
        black_value = float(black)
        white_value = float(white)
        if not math.isfinite(black_value) or not math.isfinite(white_value):
            raise ValueError(
                f"declared RAW calibration must be finite, got black={black_value} "
                f"white={white_value}"
            )
        if abs(white_value - black_value) < 1e-12:
            raise ValueError(
                f"declared RAW calibration has an empty range: black={black_value} "
                f"white={white_value}"
            )
        return {"black": black_value, "white": white_value, "mode": "declared_affine"}

    def _metadata_availability(
        self,
        metadata: dict[str, Any],
        timestamp_available: bool,
        calibration: dict[str, float] | None,
    ) -> dict[str, Any]:
        available = {
            "timestamp": bool(timestamp_available),
            "timestamp_source": "declared" if timestamp_available else "frame_index",
            "saturation_level": self.protocol.camera_dtype == "uint8"
            or self.saturated_level is not None
            or calibration is not None,
            "saturation_level_value": (
                255.0
                if self.protocol.camera_dtype == "uint8" and self.saturated_level is None
                else self.saturated_level
            ),
            "bad_pixel_mask": False,
            "exposure": False,
            "gain": False,
            "camera_intrinsics": False,
            "camera_transform": False,
            "lighting_domain": metadata.get("lighting_domain") not in (None, ""),
            "task_description": bool(metadata.get("task_description") or metadata.get("task_name")),
            "source_group": any(
                metadata.get(key) not in (None, "") for key in IDENTITY_METADATA_KEYS
            ),
            "declared_calibration": calibration is not None,
            "noise_model": self.noise_model is not None,
            "executed_action": False,
        }
        available["unavailable_reason"] = (
            "the author NPZ cache stores camera measurements, robot state and action commands "
            "only; exposure/gain/intrinsics/camera pose/defect maps and measured executed "
            "actions are not part of it, and nothing is filled in from a guess"
        )
        return available

    def _load_episode(self, path: str) -> dict[str, np.ndarray]:
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            self._cached_path = path
            self._episode_cache = {path: arrays}
            self._check_provenance(path, arrays)
        return self._episode_cache[path]

    def _check_provenance(self, path: str, arrays: dict[str, np.ndarray]) -> None:
        metadata = self._episode_metadata(path, arrays)
        declared = bool(metadata.get("synthetic_fixture", False))
        if declared:
            self._fixture_paths.add(path)
        if path in self._provenance_checked or self.fixture_provenance == "auto":
            self._provenance_checked.add(path)
            return
        self._provenance_checked.add(path)
        if self.fixture_provenance == "synthetic" and not declared:
            raise ValueError(
                f"fixture_provenance='synthetic' but {path} does not declare "
                "metadata_json['synthetic_fixture']=true"
            )
        if self.fixture_provenance == "real" and declared:
            raise ValueError(
                f"fixture_provenance='real' but {path} declares a synthetic fixture; refusing "
                "to present synthetic data as real RawVLA data"
            )

    def _episode_metadata(self, path: str, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        raw = arrays.get("metadata_json")
        if raw is None:
            return {}
        try:
            value = raw.item() if raw.ndim == 0 else raw.reshape(-1)[0]
        except (AttributeError, ValueError, IndexError):
            return {}
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json in {path} is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"metadata_json in {path} must decode to an object")
        return parsed

    def _episode_key(self, path: str) -> str:
        try:
            return Path(path).relative_to(self.raw_cache_root).as_posix()
        except ValueError:
            return Path(path).as_posix()

    def _split_of(self, path: str) -> str:
        return self._split_of_path.get(path, "unassigned")

    # -- reporting ---------------------------------------------------------------

    def missing_assets(self) -> list[dict[str, Any]]:
        """Every required/optional asset with its resolved path and state."""
        return [
            item.as_dict()
            for item in rawvla_asset_requirements(
                root=self.root,
                dataset_format=self.protocol.name,
                raw_cache_root=self.raw_cache_root,
                stats_path=self.stats_path,
                rgb_cache_root=self.rgb_cache_root,
                requires_stats=self.normalization != "none",
            )
        ]

    def spec(self) -> dict[str, Any]:
        """Data inventory, provenance, identity fields and declared numeric policy."""
        camera_ids = [self.protocol.camera_id(key) for key in self.camera_keys]
        return {
            "loader": "owac.raw.rawvla_sequence.RawVLASequenceDataset",
            "dataset_format": self.protocol.name,
            "root": str(self.root),
            "raw_cache_root": str(self.raw_cache_root),
            "rgb_cache_root": str(self.rgb_cache_root),
            "stats_path": str(self.stats_path),
            "stats_available": self.stats_path.is_file(),
            "statistics": self.statistics.as_dict(),
            "normalization": self.normalization,
            "camera_keys": list(self.camera_keys),
            "camera_ids": camera_ids,
            "camera_dtype": self.protocol.camera_dtype,
            "raw_value_policy": {
                "mode": self.protocol.raw_scaling,
                "unit": self.protocol.raw_unit,
                "evidence": self.protocol.raw_range_evidence,
                "forbidden": [
                    "dividing RoboTwin float32 RAW by 255",
                    "per-frame max normalisation of any RAW array",
                    "renaming three-channel pseudo-RAW as Bayer CFA",
                ],
            },
            "channels_per_camera": 3,
            "sampling_geometry": "per_pixel_trichromatic_collocated",
            "supported_input_modes": {
                "rawvla_rgb3": "this loader: co-located three-channel measurement windows",
                "bayer_cfa": (
                    "owac.raw.cfa plane stacks wrapped by measurement_window_from_cfa; the "
                    "RawVLA NPZ cache itself is three-channel and is never called Bayer"
                ),
                "sequence1d": "owned by the one-dimensional adapter, not by this loader",
            },
            "state_source_key": self.state_source_key,
            "state_dim": self.protocol.state_dim,
            "state_kind": self.protocol.state_kind,
            "state_note": self.protocol.state_note,
            "proprio_available": self.protocol.proprio_available,
            "executed_action_available": False,
            "executed_action_note": (
                "the cache holds commands and future supervision; no measured executed action "
                "is recorded, so the flag is False and executed_action stays None"
            ),
            "action_dim": self.protocol.action_dim,
            "action_offset": self.action_offset,
            "action_horizon": self.action_horizon,
            "window": self.window,
            "observation_stride": self.observation_stride,
            "observation_stride_verified": self.protocol.observation_stride_verified,
            "time_unit_observed": dict(sorted(self._time_units.items())),
            "timestamp_keys_checked": list(TIMESTAMP_KEYS),
            "time_unit_policy": (
                "seconds when the episode declares a finite, monotone timestamp array, "
                "otherwise 'frame' with the frame index in the timestamp field; seconds are "
                "never invented and no nominal frame rate is assumed"
            ),
            "noise_model": None if self.noise_model is None else self.noise_model.as_dict(),
            "episodes": len(self.episodes),
            "episodes_in_split": len(self.paths),
            "split": {
                "split_version": self.split.get("split_version"),
                "requested": self.split_name,
                "counts": self.split.get("counts"),
                "rule": self.split.get("notes", []),
                "manifest_path": (
                    None if self.split_manifest_path is None else str(self.split_manifest_path)
                ),
                "source_groups": len(self.split.get("source_groups", [])),
            },
            "identity_fields": self.split.get("identity_fields"),
            "identity_fields_observed": self.observed_identity_fields(),
            "identity_fields_available": {
                "relative_path": True,
                **{
                    key: any(
                        entry.get("identity_field") == f"metadata.{key}"
                        for entry in self.split.get("source_groups", [])
                    )
                    for key in IDENTITY_METADATA_KEYS
                },
                "lighting_domain": any(
                    entry.get("lighting_domains") for entry in self.split.get("source_groups", [])
                ),
            },
            "official_closed_loop_manifest": {
                "used": False,
                "note": (
                    "the official RawVLA closed-loop manifest is kept separate and never mixed "
                    "into this loader's split"
                ),
            },
            "data_present": bool(self.episodes),
            "real_data_available": self._real_data_available(),
            "real_data_available_note": (
                "True means the provenance scan covered every episode and found no synthetic "
                "declaration; None means the scan has not covered the inventory on this machine, "
                "so availability is unverified rather than claimed"
            ),
            "synthetic_fixture_episodes": sorted(self._fixture_paths),
            "provenance_scanned_episodes": len(self._provenance_checked),
            "fixture_provenance": self.fixture_provenance,
            "range_evidence": (
                None
                if self.range_evidence_path is None
                else recorded_range_evidence(self.range_evidence_path)
            ),
            "missing_assets": self.missing_assets(),
        }

    def stats(self, *, max_files: int = 4) -> dict[str, Any]:
        """``spec()`` plus measured ranges and provenance from a bounded file scan.

        The scans run before ``spec()`` so the reported provenance and
        real-data availability already reflect the scanned episodes.
        """
        ranges = self.measure_ranges(max_files=max_files)
        provenance = self.scan_provenance(max_files=max_files)
        report = self.spec()
        report["ranges"] = ranges
        report["provenance"] = provenance
        report["ranges"]["note"] = (
            report["ranges"]["note"] + " Provenance of the same sample is reported separately."
        )
        return report

    def scan_provenance(self, *, max_files: int | None = None) -> dict[str, Any]:
        """Read identity/provenance metadata of a bounded episode sample.

        Only ``metadata_json`` is decompressed per episode, so this stays cheap
        enough for a manifest check.  It never touches the camera arrays.
        """
        limit = len(self.episodes) if max_files is None else max(0, int(max_files))
        scanned: list[dict[str, Any]] = []
        for path in self.episodes[:limit]:
            with np.load(path, allow_pickle=False) as archive:
                arrays = (
                    {"metadata_json": np.asarray(archive["metadata_json"])}
                    if "metadata_json" in archive.files
                    else {}
                )
            metadata = self._episode_metadata(path, arrays)
            self._scanned_metadata[path] = metadata
            self._check_provenance(path, arrays)
            scanned.append(
                {
                    "episode": self._episode_key(path),
                    "synthetic_fixture": bool(metadata.get("synthetic_fixture", False)),
                    "lighting_domain": metadata.get("lighting_domain"),
                    "source_identity": {
                        key: metadata.get(key)
                        for key in IDENTITY_METADATA_KEYS
                        if metadata.get(key)
                    },
                    "task_description": str(
                        metadata.get("task_description") or metadata.get("task_name") or ""
                    ),
                }
            )
        declared = sum(1 for item in scanned if item["synthetic_fixture"])
        return {
            "episodes_scanned": len(scanned),
            "episodes_total": len(self.episodes),
            "complete": len(scanned) == len(self.episodes),
            "synthetic_fixture_declared": declared,
            "synthetic_fixture_all_scanned": bool(scanned) and declared == len(scanned),
            "episodes": scanned,
            "note": (
                "a fixture must declare metadata_json['synthetic_fixture']=true; synthetic "
                "fixtures are never reported as real RawVLA data"
            ),
        }

    def measure_ranges(self, *, max_files: int = 4) -> dict[str, Any]:
        """Measured dtype/shape/min/max over the first ``max_files`` episodes.

        This is an explicit bounded sample, not a full scan of the inventory; the
        returned ``complete`` flag says which of the two it was.
        """
        keys = list(self.protocol.camera_keys)
        for extra in (self.state_source_key, "actions"):
            if extra not in keys:
                keys.append(extra)
        ranges: dict[str, dict[str, Any]] = {}
        scanned = 0
        for path in self.episodes[: max(0, int(max_files))]:
            with np.load(path, allow_pickle=False) as archive:
                files = set(archive.files)
                for key in keys:
                    if key not in files:
                        continue
                    array = np.asarray(archive[key])
                    entry = ranges.setdefault(
                        key,
                        {
                            "dtype": str(array.dtype),
                            "shape": list(array.shape),
                            "min": math.inf,
                            "max": -math.inf,
                            "finite": True,
                            "files": 0,
                            "episodes": [],
                        },
                    )
                    entry["min"] = min(entry["min"], float(np.min(array)))
                    entry["max"] = max(entry["max"], float(np.max(array)))
                    entry["files"] += 1
                    entry["episodes"].append(self._episode_key(path))
                    if array.dtype.kind == "f":
                        entry["finite"] = bool(entry["finite"] and np.isfinite(array).all())
            scanned += 1
        for entry in ranges.values():
            entry["min"] = None if math.isinf(entry["min"]) else entry["min"]
            entry["max"] = None if math.isinf(entry["max"]) else entry["max"]
        return {
            "keys": ranges,
            "files_scanned": scanned,
            "files_total": len(self.episodes),
            "complete": scanned == len(self.episodes),
            "note": (
                "bounded sample scan of the split inventory; call with max_files=len(dataset) "
                "for a full scan. Values are stored digital numbers before the load-time "
                "scaling declared in spec()['raw_value_policy']."
            ),
        }

    def observed_identity_fields(self) -> dict[str, list[str]]:
        """Identity values actually observed in the scanned episode metadata."""
        observed: dict[str, set[str]] = {"lighting_domain": set()}
        for key in IDENTITY_METADATA_KEYS:
            observed[key] = set()
        for metadata in self._scanned_metadata.values():
            domain = metadata.get("lighting_domain")
            if domain not in (None, ""):
                observed["lighting_domain"].add(str(domain))
            for key in IDENTITY_METADATA_KEYS:
                value = metadata.get(key)
                if value not in (None, ""):
                    observed[key].add(str(value))
        return {key: sorted(values) for key, values in observed.items()}

    def rebuild_split(self, *, use_metadata: bool = True, split: str | None = None) -> dict:
        """Rebuild the split, optionally grouping by scanned metadata identity fields.

        Availability is checked before use: when ``use_metadata`` is set and the
        provenance scan has not run yet, it is run over the whole inventory so a
        declared capture session or acquisition id can pull every file of that
        source onto one side.  Path-based grouping is always the fallback.
        """
        if use_metadata and len(self._scanned_metadata) < len(self.episodes):
            self.scan_provenance(max_files=None)
        metadata_by_path = (
            {path: self._scanned_metadata.get(path, {}) for path in self.episodes}
            if use_metadata
            else None
        )
        self.split = build_split_manifest(
            self.episodes,
            dataset_format=self.protocol.name,
            relative_to=self.raw_cache_root,
            train_fraction=self.train_fraction,
            split_version=self.split_version,
            metadata_by_path=metadata_by_path,
        )
        self._split_of_path = {}
        for side, files in self.split.get("sides", {}).items():
            for entry in files:
                self._split_of_path[str(entry)] = side
        self.paths = self._paths_for_split(split or self.split_name)
        self.virtual_length = len(self.paths) * self.samples_per_episode
        return self.split

    def _real_data_available(self) -> bool | None:
        """``False``/``True`` only when the provenance scan backs it, else ``None``."""
        if self.fixture_provenance == "synthetic":
            return False
        if self._fixture_paths:
            return len(self._fixture_paths) < len(self.episodes)
        if self.episodes and len(self._provenance_checked) >= len(self.episodes):
            return True
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RawVLASequenceDataset(dataset_format={self.protocol.name!r}, "
            f"episodes={len(self.paths)}, window={self.window}, "
            f"stride={self.observation_stride}, action_horizon={self.action_horizon})"
        )


def recorded_range_evidence(path: str | Path) -> dict[str, Any]:
    """Read a recorded RawVLA dataset check and expose its measured ranges.

    The RawVLA-Bench snapshot is not on this machine, so the only locally
    auditable range evidence is the committed deployment record
    (``research/experiments/results/2026-10-10_rawvla_pi05_deployment/dataset-check.json``).
    It is reported verbatim with its revision and is never presented as a
    measurement taken by this loader.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"recorded range evidence not found: {target}")
    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)
    samples: list[dict[str, Any]] = []
    for sample in data.get("samples", []):
        features = sample.get("features", {})
        samples.append(
            {
                "path": sample.get("path"),
                "sha256": sample.get("sha256"),
                "features": {
                    key: {
                        "dtype": value.get("dtype"),
                        "shape": value.get("shape"),
                        "min": value.get("min"),
                        "max": value.get("max"),
                        "finite": value.get("finite"),
                    }
                    for key, value in features.items()
                },
            }
        )
    return {
        "source": str(target),
        "revision": data.get("revision"),
        "files": data.get("files"),
        "bytes": data.get("bytes"),
        "samples": samples,
        "note": (
            "recorded by the 2026-10-10 server deployment check, not re-measured on this "
            "machine; the RawVLA-Bench snapshot is absent locally"
        ),
    }


def iter_windows(samples: Iterable[SequenceSample]) -> Iterator[MeasurementWindow]:
    """Yield every camera window of every sample in order; a small streaming helper."""
    for sample in samples:
        for camera_id in sorted(sample.windows):
            yield sample.windows[camera_id]

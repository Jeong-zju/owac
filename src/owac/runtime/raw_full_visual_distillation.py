"""Two-GPU full-student visual-head RAW token distillation.

The teacher and the pi05 language/action backend are never constructed,
optimized, or stored by this module.  The student visual head is built directly
from the actual pi05 safetensors vision/projector tensors; the RAW stem and
history gate are fresh with seed 101.  All visual parameters are FP32 trainable
and forward computation uses BF16 autocast; optimizer states stay FP32.  The
causal 6-frame packet, full-prefix burn-in, 6-endpoint unroll and token loss are
the existing temporal student semantics, reused through its trainer forward.

Only the visual head is trainable: stem, gate, position embedding, all SigLIP
encoder blocks, post-layernorm and the original projector (428,579,952 tensors).
No action loss, closed-loop episode, B/D arm or download exists here.

The module is intentionally standalone: it does not import the teacher policy or
the action backend.  It expects the immutable teacher-token cache produced by the
accepted local stage and verifies its identity without rewriting the manifest.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import mmap
import os
import re
import signal
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from owac.runtime.raw_full_visual_fastdata import (
    CachedPacketDataset,
    PerformanceOptions,
    load_performance_options,
    preload_raw_episodes,
)

__all__ = [
    "EXPECTED_FULL_VISUAL_PARAMETERS",
    "FullVisualBudget",
    "PathMapper",
    "build_full_visual_student",
    "classify_full_visual_parameter",
    "deterministic_source_split",
    "global_accumulation_weights",
    "plan_checkpoint_retention",
    "select_full_visual_trainable",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_FULL_VISUAL_PARAMETERS = 428_579_952
DEFAULT_WORLD_SIZE = 2
EXPECTED_EFFECTIVE_BATCH = 16
CACHE_SCHEMA = "owac.raw-stem-temporal.cache.v2"
CHECKPOINT_SCHEMA = "owac.raw-full-visual.checkpoint.v1"
PREFLIGHT_SCHEMA = "owac.raw-full-visual.preflight.v1"
RANK_LOCAL_CUDA_RNG_SCHEMA = "owac.raw-full-visual.rank-local-cuda-rng.v1"
FULL_VISUAL_SCOPE = "full_visual_head"
FULL_VISUAL_PREFIXES = (
    "gate_logit",
    "base.vision_tower.vision_model.embeddings.patch_embedding.",
    "base.vision_tower.vision_model.embeddings.position_embedding.",
    "base.vision_tower.vision_model.encoder.layers.",
    "base.vision_tower.vision_model.post_layernorm.",
    "base.projector.",
)
PARENT_UPDATE_EQUIVALENTS_SPENT = 1884
TOTAL_UPDATE_EQUIVALENTS = 5000
PARENT_ANCHORS_EQUIVALENTS_SPENT = 30144
TOTAL_ANCHOR_EQUIVALENTS = 80000
PRIOR_GPU_WALL_SECONDS = 89558.73680601672
TOTAL_GPU_WALL_SECONDS = 230400.0
PRIOR_OUTPUT_BYTES = 6078252068
TOTAL_OUTPUT_BYTES = 21474836480


SHARED_PROTOCOL = "owac.shared-rawvla.task6-normal.8-2-2.v1"
LEGACY_PROTOCOL = "legacy-c-full-40-5-5.v1"


def _data_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(config.get("data") or {})
    protocol = str(data.get("protocol") or LEGACY_PROTOCOL)
    if protocol == SHARED_PROTOCOL:
        defaults = {
            "expected_source_count": 12,
            "expected_splits": {"train": 8, "validation": 2, "test": 2},
            "expected_train_anchors": 118,
            "expected_validation_windows": 46,
            "expected_cache_rows": 204,
        }
    else:
        defaults = {
            "expected_source_count": 50,
            "expected_splits": {"train": 40, "validation": 5, "test": 5},
            "expected_train_anchors": 667,
            "expected_validation_windows": 102,
            "expected_cache_rows": 969,
        }
    result = {
        "protocol": protocol,
        "expected_source_count": int(
            data.get("expected_source_count", defaults["expected_source_count"])
        ),
        "expected_splits": {
            key: int(value)
            for key, value in dict(data.get("expected_splits", defaults["expected_splits"])).items()
        },
        "expected_train_anchors": int(
            data.get("expected_train_anchors", defaults["expected_train_anchors"])
        ),
        "expected_validation_windows": int(
            data.get("expected_validation_windows", defaults["expected_validation_windows"])
        ),
        "expected_cache_rows": int(
            data.get("expected_cache_rows", defaults["expected_cache_rows"])
        ),
        "test_targets_generated": bool(data.get("test_targets_generated", False)),
    }
    if set(result["expected_splits"]) != {"train", "validation", "test"}:
        raise FullVisualError("data split contract must name train/validation/test")
    if result["expected_splits"] != defaults["expected_splits"] and protocol == LEGACY_PROTOCOL:
        raise FullVisualError("legacy data split contract is frozen 40/5/5")
    return result


class FullVisualError(RuntimeError):
    """A real full-visual training contract or resource is unavailable."""


# --------------------------------------------------------------------------- #
# Generic durable helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fsync_dir(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    _fsync_dir(target.parent)
    return target


def _append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _proc_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 1 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def owned_run_bytes(root: str | Path) -> int:
    """Unique owned bytes: exclude symlinks and deduplicate hardlinks by inode."""
    root_path = Path(root)
    if not root_path.exists():
        return 0
    total = 0
    seen: set[tuple[int, int]] = set()
    for directory, dirnames, filenames in os.walk(root_path, followlinks=False):
        current = Path(directory)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            key = (int(stat.st_dev), int(stat.st_ino))
            if key in seen:
                continue
            seen.add(key)
            total += int(stat.st_size)
    return total


# --------------------------------------------------------------------------- #
# Budget and split/DP accounting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class FullVisualBudget:
    prior_gpu_wall_seconds: float = PRIOR_GPU_WALL_SECONDS
    total_gpu_wall_seconds: float = TOTAL_GPU_WALL_SECONDS
    prior_output_bytes: int = PRIOR_OUTPUT_BYTES
    total_output_bytes: int = TOTAL_OUTPUT_BYTES
    spent_update_equivalents: int = PARENT_UPDATE_EQUIVALENTS_SPENT
    total_update_equivalents: int = TOTAL_UPDATE_EQUIVALENTS
    spent_anchor_equivalents: int = PARENT_ANCHORS_EQUIVALENTS_SPENT
    total_anchor_equivalents: int = TOTAL_ANCHOR_EQUIVALENTS
    new_updates_cap: int = 3116
    new_anchors_cap: int = 49856

    @property
    def remote_gpu_wall_cap_seconds(self) -> float:
        value = self.total_gpu_wall_seconds - self.prior_gpu_wall_seconds
        if value <= 0:
            raise FullVisualError("remote GPU wall budget is exhausted")
        return value

    @property
    def remote_output_cap_bytes(self) -> int:
        value = self.total_output_bytes - self.prior_output_bytes
        if value <= 0:
            raise FullVisualError("remote output budget is exhausted")
        return value

    @property
    def remaining_update_equivalents(self) -> int:
        return self.total_update_equivalents - self.spent_update_equivalents

    @property
    def remaining_anchor_equivalents(self) -> int:
        return self.total_anchor_equivalents - self.spent_anchor_equivalents

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FullVisualBudget:
        remote = dict(config.get("remote") or {})
        training = dict(config.get("training") or {})
        prior = dict(config.get("prior_stage") or {})
        budget = cls(
            prior_gpu_wall_seconds=float(prior.get("gpu_wall_seconds", PRIOR_GPU_WALL_SECONDS)),
            total_gpu_wall_seconds=float(
                remote.get("total_gpu_wall_seconds", TOTAL_GPU_WALL_SECONDS)
            ),
            prior_output_bytes=int(prior.get("output_bytes", PRIOR_OUTPUT_BYTES)),
            total_output_bytes=int(remote.get("total_output_bytes", TOTAL_OUTPUT_BYTES)),
            spent_update_equivalents=int(
                prior.get("spent_update_equivalents", PARENT_UPDATE_EQUIVALENTS_SPENT)
            ),
            total_update_equivalents=int(
                training.get("total_update_equivalents", TOTAL_UPDATE_EQUIVALENTS)
            ),
            spent_anchor_equivalents=int(
                prior.get("spent_anchor_equivalents", PARENT_ANCHORS_EQUIVALENTS_SPENT)
            ),
            total_anchor_equivalents=int(
                training.get("total_anchor_equivalents", TOTAL_ANCHOR_EQUIVALENTS)
            ),
            new_updates_cap=int(training.get("max_new_global_updates", 3116)),
            new_anchors_cap=int(training.get("max_new_anchor_presentations", 49856)),
        )
        if budget.new_updates_cap != budget.remaining_update_equivalents:
            raise FullVisualError("new update cap differs from remaining update equivalents")
        if budget.new_anchors_cap != budget.remaining_anchor_equivalents:
            raise FullVisualError("new anchor cap differs from remaining anchor equivalents")
        if budget.new_updates_cap != 3116 or budget.new_anchors_cap != 49856:
            raise FullVisualError("the authorized remaining update/anchor ceilings differ")
        if abs(budget.prior_gpu_wall_seconds - PRIOR_GPU_WALL_SECONDS) > 1e-9:
            raise FullVisualError("prior GPU wall debit differs from the frozen receipt")
        if budget.prior_output_bytes != PRIOR_OUTPUT_BYTES:
            raise FullVisualError("prior output debit differs from the frozen receipt")
        return budget

    def assert_gpu_usage(self, gpu_wall_sum_seconds: float) -> None:
        if not math.isfinite(gpu_wall_sum_seconds) or gpu_wall_sum_seconds < 0:
            raise FullVisualError("GPU wall usage must be finite and non-negative")
        if gpu_wall_sum_seconds > self.remote_gpu_wall_cap_seconds + 1e-6:
            raise FullVisualError(
                f"remote GPU wall {gpu_wall_sum_seconds:.3f}s exceeds cap "
                f"{self.remote_gpu_wall_cap_seconds:.3f}s"
            )


def classify_full_visual_parameter(name: str) -> str | None:
    """Map one student parameter name to the authorized full-visual category."""
    if name == "gate_logit":
        return "gate"
    if ".embeddings.patch_embedding." in name:
        return "stem"
    if ".embeddings.position_embedding." in name:
        return "position"
    if ".encoder.layers." in name:
        return "blocks"
    if ".post_layernorm." in name:
        return "norm"
    if ".projector." in name:
        return "projector"
    return None


def select_full_visual_trainable(named_parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Select exactly the full visual head; return counts and rejected names."""
    selected: list[str] = []
    rejected: list[str] = []
    categories = {key: 0 for key in ("gate", "stem", "position", "blocks", "norm", "projector")}
    total = 0
    for name, parameter in named_parameters.items():
        category = classify_full_visual_parameter(str(name))
        if category is None:
            rejected.append(str(name))
            continue
        selected.append(str(name))
        categories[category] += int(parameter.numel())
        total += int(parameter.numel())
    return {
        "selected": selected,
        "rejected": rejected,
        "categories": categories,
        "total": total,
        "expected": EXPECTED_FULL_VISUAL_PARAMETERS,
        "matches_expected": total == EXPECTED_FULL_VISUAL_PARAMETERS and not rejected,
    }


def global_accumulation_weights(
    local_token_counts: Sequence[int],
    *,
    world_size: int,
    global_token_total: int,
) -> list[float]:
    """Correct DDP scale: world * local_tokens/global_tokens per local anchor."""
    if world_size < 1 or global_token_total <= 0:
        raise FullVisualError("invalid DDP weighting denominator")
    weights = [
        float(world_size) * int(count) / int(global_token_total) for count in local_token_counts
    ]
    if any(not math.isfinite(item) or item <= 0 for item in weights):
        raise FullVisualError("DDP per-anchor weight is not positive/finite")
    return weights


def loss_component_numerators(components: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Sum detached FP32 loss numerators weighted by each microbatch token count."""
    mse_numerator = 0.0
    cosine_numerator = 0.0
    total_numerator = 0.0
    tokens = 0
    for item in components:
        count = int(item["tokens"])
        if count <= 0:
            raise FullVisualError("loss logging component has no tokens")
        mse_numerator += float(item["mse"]) * count
        cosine_numerator += float(item["cosine"]) * count
        total_numerator += float(item["total"]) * count
        tokens += count
    if tokens <= 0:
        raise FullVisualError("loss logging has no tokens")
    return {
        "mse_numerator": mse_numerator,
        "cosine_numerator": cosine_numerator,
        "total_numerator": total_numerator,
        "tokens": float(tokens),
    }


def finalize_global_loss_components(
    numerators: Mapping[str, Any], *, global_token_total: int
) -> dict[str, Any]:
    """Divide globally accumulated token-weighted numerators once per update."""
    denominator = int(global_token_total)
    if denominator <= 0:
        raise FullVisualError("global loss denominator is zero")
    numerator_tokens = int(float(numerators["tokens"]))
    if numerator_tokens != denominator:
        raise FullVisualError(
            f"global loss numerator tokens {numerator_tokens} != denominator {denominator}"
        )
    mse = float(numerators["mse_numerator"]) / denominator
    cosine = float(numerators["cosine_numerator"]) / denominator
    total = float(numerators["total_numerator"]) / denominator
    return {
        "mse": mse,
        "cosine": cosine,
        "total": total,
        "mse_numerator": float(numerators["mse_numerator"]),
        "cosine_numerator": float(numerators["cosine_numerator"]),
        "total_numerator": float(numerators["total_numerator"]),
        "global_token_denominator": denominator,
        "scope": "global_effective_batch_16",
    }


def rank_slice(batch: Sequence[Any], *, rank: int, world_size: int) -> list[Any]:
    """Disjoint contiguous slice of the global effective batch for one rank."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise FullVisualError("invalid rank/world_size")
    batch = list(batch)
    if len(batch) % world_size:
        raise FullVisualError("global batch is not divisible by world_size")
    width = len(batch) // world_size
    return batch[rank * width : (rank + 1) * width]


def deterministic_source_split(
    manifest: Any, expected_splits: Mapping[str, int] | None = None
) -> dict[str, tuple[str, ...]]:
    """Require the configured source split; test is never train/select."""
    expected = (
        dict(expected_splits)
        if expected_splits is not None
        else {"train": 40, "validation": 5, "test": 5}
    )
    result = {}
    for split in ("train", "validation", "test"):
        expected_count = int(expected[split])
        sources = tuple(str(item.source_demo_id) for item in manifest.sources_for_split(split))
        if len(sources) != expected_count:
            raise FullVisualError(
                f"{split} split has {len(sources)} sources, expected {expected_count}"
            )
        if len(set(sources)) != len(sources):
            raise FullVisualError(f"{split} split contains duplicate sources")
        result[split] = sources
    if set(result["train"]) & set(result["validation"]):
        raise FullVisualError("train and validation source sets overlap")
    if set(result["train"]) & set(result["test"]):
        raise FullVisualError("train and test source sets overlap")
    if set(result["validation"]) & set(result["test"]):
        raise FullVisualError("validation and test source sets overlap")
    return result


# --------------------------------------------------------------------------- #
# Path mapping and immutable cache
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MappedPath:
    logical: str
    physical: str
    mapped: bool


def _mapped_record(mapped: MappedPath, field: str) -> dict[str, Any]:
    """Serialize a slots dataclass explicitly; never rely on __dict__."""
    return {
        "field": field,
        "logical": mapped.logical,
        "physical": mapped.physical,
        "mapped": bool(mapped.mapped),
    }


class PathMapper:
    """Resolve logical absolute paths through an explicit, non-rewriting map."""

    def __init__(self, mappings: Sequence[Mapping[str, Any]] = ()) -> None:
        self.mappings = []
        for item in mappings:
            logical = str(item.get("logical_prefix") or "")
            physical = str(item.get("physical_prefix") or "")
            if logical and physical:
                self.mappings.append((logical.rstrip("/"), physical.rstrip("/")))

    def resolve(self, value: str | Path, *, must_exist: bool = True) -> MappedPath:
        logical = str(value)
        direct = Path(logical).expanduser()
        if direct.exists():
            return MappedPath(logical=logical, physical=str(direct), mapped=False)
        for logical_prefix, physical_prefix in self.mappings:
            if logical == logical_prefix or logical.startswith(logical_prefix + "/"):
                suffix = logical[len(logical_prefix) :]
                candidate = Path(physical_prefix + suffix)
                if candidate.exists():
                    return MappedPath(logical=logical, physical=str(candidate), mapped=True)
        if must_exist:
            raise FullVisualError(f"mapped asset path does not exist: {logical}")
        return MappedPath(logical=logical, physical=logical, mapped=False)

    def report(self, values: Sequence[str]) -> list[dict[str, Any]]:
        return [
            {
                "logical": item.logical,
                "physical": item.physical,
                "mapped": item.mapped,
            }
            for item in (self.resolve(value, must_exist=False) for value in values)
        ]


def _map_config_paths(
    config: Mapping[str, Any], mapper: PathMapper
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create an in-memory mapped config; never rewrite the manifest identity."""
    value = copy.deepcopy(dict(config))
    assets = value.get("assets")
    teacher = value.get("teacher_requirement")
    if not isinstance(assets, dict) or not isinstance(teacher, dict):
        raise FullVisualError("config requires assets and teacher_requirement")
    records: list[dict[str, Any]] = []
    path_fields = (
        "teacher_registry",
        "teacher_frontend_checkpoint",
        "source_manifest",
        "load_audit",
        "checkpoint",
        "tokenizer",
        "statistics",
        "cache_root",
    )
    for key in path_fields:
        raw = assets.get(key)
        if isinstance(raw, str) and raw:
            mapped = mapper.resolve(raw)
            assets[key] = mapped.physical
            records.append(_mapped_record(mapped, f"assets.{key}"))
    for key in ("author_root", "starvla_root"):
        raw = assets.get(key)
        if isinstance(raw, str) and raw:
            mapped = mapper.resolve(raw)
            assets[key] = mapped.physical
            records.append(_mapped_record(mapped, f"assets.{key}"))
    config_paths = assets.get("teacher_config_paths")
    if isinstance(config_paths, list):
        mapped_paths = []
        for raw in config_paths:
            mapped = mapper.resolve(str(raw))
            mapped_paths.append(mapped.physical)
            records.append(_mapped_record(mapped, "assets.teacher_config_paths"))
        assets["teacher_config_paths"] = mapped_paths
    output_root = value.get("remote", {}).get("output_root") or assets.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise FullVisualError("config lacks a remote output root")
    assets["output_root"] = output_root
    value["assets"] = assets
    value["teacher_requirement"] = teacher
    return value, {"records": records, "output_root": output_root}


class ImmutableTargetCache:
    """Read the accepted teacher cache without recomputing path-dependent identity."""

    def __init__(self, root: str | Path, *, expected_identity_key: str) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FullVisualError("teacher cache manifest is absent")
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, Mapping) or manifest.get("schema") != CACHE_SCHEMA:
            raise FullVisualError("teacher cache manifest schema differs")
        # The manifest directory key is an opaque compatibility path.  Legacy
        # identity/hash fields are retained for readers but not compared.
        key = manifest.get("identity_key")
        if (
            not isinstance(key, str)
            or not key
            or key in {".", ".."}
            or Path(key).name != key
            or "/" in key
            or "\\" in key
        ):
            raise FullVisualError(
                "teacher cache manifest identity must be a safe directory component"
            )
        self.identity_key = key
        self.identity = dict(manifest.get("identity") or {})
        self.manifest = manifest
        self._rows: dict[tuple[str, int], Any] = {}
        self.preload_report: dict[str, Any] | None = None

    @staticmethod
    def _source_slug(source_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id)

    def row_path(self, source_id: str, endpoint: int) -> Path:
        return (
            self.root / self.identity_key / self._source_slug(source_id) / f"{int(endpoint):06d}.pt"
        )

    def exists(self, source_id: str, endpoint: int) -> bool:
        return self.row_path(source_id, endpoint).is_file()

    def cached_endpoints(self, source_id: str) -> tuple[int, ...]:
        directory = self.root / self.identity_key / self._source_slug(source_id)
        if not directory.is_dir():
            return ()
        values = []
        for path in sorted(directory.glob("*.pt")):
            try:
                values.append(int(path.stem))
            except ValueError:
                continue
        return tuple(values)

    def preload(self, source_ids: Sequence[str]) -> dict[str, Any]:
        """Verify and memoize only train+validation rows, once per rank."""
        import resource

        started = time.monotonic()
        before_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        rows = 0
        bytes_loaded = 0
        for source_id in source_ids:
            for endpoint in self.cached_endpoints(source_id):
                path = self.row_path(source_id, endpoint)
                bytes_loaded += int(path.stat().st_size)
                self.read(source_id, endpoint)
                rows += 1
        elapsed = time.monotonic() - started
        after_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        self.preload_report = {
            "rows": rows,
            "sources": len(source_ids),
            "file_bytes": bytes_loaded,
            "load_seconds": elapsed,
            "rss_before_bytes": before_rss,
            "rss_after_bytes": after_rss,
            "memoized": True,
            "scope": "train_and_validation_only",
            "test_rows_preloaded": False,
        }
        return dict(self.preload_report)

    def read(self, source_id: str, endpoint: int) -> Any:
        import torch

        from owac.raw.temporal_distillation import TemporalTargetRow

        key = (str(source_id), int(endpoint))
        if key in self._rows:
            return self._rows[key]
        path = self.row_path(source_id, endpoint)
        if not path.is_file():
            raise FullVisualError(f"teacher cache row is absent: {source_id}@{endpoint}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("schema") != CACHE_SCHEMA:
            raise FullVisualError("teacher cache row schema differs")
        if str(payload.get("source_id")) != str(source_id) or int(
            payload.get("endpoint", -1)
        ) != int(endpoint):
            raise FullVisualError("teacher cache row key differs")
        # Legacy payload_sha256 fields are intentionally ignored; tensors and
        # structural row fields are validated by TemporalTargetRow.from_payload.
        row = TemporalTargetRow.from_payload(payload.get("row"))
        if row.source_id != source_id or int(row.endpoint) != int(endpoint):
            raise FullVisualError("teacher cache row semantics differ")
        self._rows[key] = row
        return row


# --------------------------------------------------------------------------- #
# Safetensors subset loader and standalone pi05 visual student
# --------------------------------------------------------------------------- #
_SAFETENSORS_DTYPES = {
    "F32": "float32",
    "BF16": "bfloat16",
    "F16": "float16",
    "F64": "float64",
}


def parse_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise FullVisualError("safetensors file is truncated")
        header_length = int.from_bytes(raw, "little")
        if header_length <= 0 or header_length > 256 * 1024 * 1024:
            raise FullVisualError("safetensors header length is invalid")
        header = json.loads(stream.read(header_length))
    if not isinstance(header, dict):
        raise FullVisualError("safetensors header is not an object")
    return header, 8 + header_length


def load_safetensors_subset(
    path: str | Path,
    *,
    key_map: Mapping[str, str],
    dtype: Any,
) -> dict[str, Any]:
    """Memory-map only selected safetensors tensors; no full backend allocation."""
    import torch

    header, data_start = parse_safetensors_header(path)
    missing = [key for key in key_map if key not in header]
    if missing:
        raise FullVisualError(f"safetensors lacks selected tensors: {missing[:5]}")
    result: dict[str, Any] = {}
    with Path(path).open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for source_key, target_key in key_map.items():
                item = header[source_key]
                source_dtype = _SAFETENSORS_DTYPES.get(str(item.get("dtype")))
                if source_dtype is None:
                    raise FullVisualError(f"unsupported safetensors dtype: {item.get('dtype')}")
                shape = tuple(int(v) for v in item.get("shape", ()))
                offsets = item.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2:
                    raise FullVisualError("safetensors offsets are invalid")
                start, end = int(offsets[0]), int(offsets[1])
                if start < 0 or end < start:
                    raise FullVisualError("safetensors offset range is invalid")
                element = torch.empty((), dtype=getattr(torch, source_dtype)).element_size()
                numel = math.prod(shape) if shape else 1
                if end - start != numel * element:
                    raise FullVisualError("safetensors tensor byte length differs")
                tensor = torch.frombuffer(
                    mapped,
                    dtype=getattr(torch, source_dtype),
                    count=numel,
                    offset=data_start + start,
                ).reshape(shape)
                result[target_key] = tensor.to(dtype=dtype).clone()
    return result


def _vision_architecture(config: Mapping[str, Any]) -> dict[str, int]:
    student = dict(config.get("student") or {})
    value = dict(student.get("vision_architecture") or {})
    required = {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "image_size": 224,
        "patch_size": 14,
        "projection_dim": 2048,
        "num_channels": 3,
    }
    for key, default in required.items():
        value[key] = int(value.get(key, default))
    return value


def build_full_visual_student(
    config: Mapping[str, Any],
    *,
    device: str = "cuda",
    dtype: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Standalone student from actual pi05 vision/projector safetensors."""
    import torch

    from owac.baselines.raw_stem_siglip import (
        InitializationAudit,
        LocalConvStem,
        RawStemSiglipStudent,
    )
    from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent

    target_dtype = dtype or torch.float32
    assets = dict(config.get("assets") or {})
    student_cfg = dict(config.get("student") or {})
    starvla_root = Path(str(assets.get("starvla_root") or ""))
    if not starvla_root.is_dir():
        raise FullVisualError("starvla source root is absent for standalone vision build")
    if str(starvla_root) not in sys.path:
        sys.path.insert(0, str(starvla_root))
    try:
        from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import (
            SiglipVisionModel,
        )
        from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIMultiModalProjector
        from transformers.models.auto import CONFIG_MAPPING
    except Exception as exc:
        raise FullVisualError(
            "standalone pi05 vision dependencies are unavailable "
            f"(transformers/StarVLA vendored SigLIP): {exc}"
        ) from exc

    arch = _vision_architecture(config)
    hf_config = CONFIG_MAPPING["paligemma"]()
    vision_config = hf_config.vision_config
    for name, value in arch.items():
        if name == "projection_dim":
            continue
        setattr(vision_config, name, value)
    vision_config.projection_dim = arch["projection_dim"]
    vision_config.projector_hidden_act = "gelu_fast"
    vision_config.torch_dtype = "float32"
    # Match the inspected source execution semantics, not merely the weights.
    vision_config._attn_implementation = "eager"
    vision_config.hidden_act = "gelu_pytorch_tanh"
    vision_config.layer_norm_eps = 1e-6
    vision_tower = SiglipVisionModel(vision_config).to(dtype=target_dtype)
    vision_tower.vision_model.config._attn_implementation = "eager"
    projector = OpenPIMultiModalProjector(hf_config).to(dtype=target_dtype)

    checkpoint = Path(str(assets.get("checkpoint")))
    if not checkpoint.is_file():
        raise FullVisualError(f"pi05 checkpoint is absent: {checkpoint}")
    # Active build deliberately does not hash checkpoint files.  Torch state
    # load and structural tensor/shape checks below are the active checks.

    vision_prefix = "vlm.model.model.vision_tower."
    projector_prefix = "vlm.model.model.multi_modal_projector."
    header, _ = parse_safetensors_header(checkpoint)
    vision_keys = {
        key: key[len(vision_prefix) :] for key in header if key.startswith(vision_prefix)
    }
    projector_keys = {
        key: key[len(projector_prefix) :] for key in header if key.startswith(projector_prefix)
    }
    if not vision_keys or not projector_keys:
        raise FullVisualError("pi05 safetensors lacks vision/projector tensors")
    vision_state = load_safetensors_subset(checkpoint, key_map=vision_keys, dtype=target_dtype)
    projector_state = load_safetensors_subset(
        checkpoint, key_map=projector_keys, dtype=target_dtype
    )
    missing, unexpected = vision_tower.load_state_dict(vision_state, strict=False)
    missing_projector, unexpected_projector = projector.load_state_dict(
        projector_state, strict=False
    )
    if missing or unexpected or missing_projector or unexpected_projector:
        raise FullVisualError(
            "pi05 visual state did not load strictly: "
            f"vision missing={list(missing)[:5]} unexpected={list(unexpected)[:5]} "
            f"projector missing={list(missing_projector)[:5]} "
            f"unexpected={list(unexpected_projector)[:5]}"
        )

    geometry = __import__(
        "owac.baselines.raw_stem_siglip", fromlist=["VisionGeometry"]
    ).VisionGeometry.from_modules(vision_tower, projector)
    patch_stride = geometry.packed_patch_stride(str(student_cfg.get("input_mode", "rawvla_rgb3")))
    stem = LocalConvStem(
        int(arch["num_channels"]),
        int(arch["hidden_size"]),
        patch_stride,
        conv_channels=int(student_cfg.get("stem_hidden", 64)),
        seed=int(student_cfg.get("stem_seed", 101)),
        dtype=target_dtype,
        device="cpu",
    )
    # Map the standalone subset back to the actual pi05 source keys and verify
    # every inherited value independently after the BF16 -> FP32 cast.
    patch_prefix = "vision_model.embeddings.patch_embedding."
    inherited_vision_keys = {
        key: value for key, value in vision_state.items() if not key.startswith(patch_prefix)
    }
    source_module_state = {**inherited_vision_keys, **projector_state}
    student_state = {
        **dict(vision_tower.state_dict()),
        **dict(projector.state_dict()),
    }
    expected_inherited = 437
    if len(source_module_state) != expected_inherited:
        raise FullVisualError(
            f"standalone inherited tensor count {len(source_module_state)} != {expected_inherited}"
        )
    inherited_mapping: list[dict[str, Any]] = []
    for module_key, source_tensor in sorted(source_module_state.items()):
        if module_key not in student_state:
            raise FullVisualError(f"student lacks inherited source tensor: {module_key}")
        student_tensor = student_state[module_key]
        if tuple(student_tensor.shape) != tuple(source_tensor.shape):
            raise FullVisualError(f"inherited tensor shape differs: {module_key}")
        if not bool(
            torch.equal(student_tensor.detach().cpu().float(), source_tensor.detach().cpu().float())
        ):
            raise FullVisualError(f"inherited tensor value differs after cast: {module_key}")
        if module_key.startswith("vision_model."):
            source_key = vision_prefix + module_key
        else:
            source_key = projector_prefix + module_key
        inherited_mapping.append(
            {
                "source_key": source_key,
                "module_key": module_key,
                "shape": list(source_tensor.shape),
                "source_dtype": str(source_tensor.dtype).replace("torch.", ""),
                "student_dtype": str(student_tensor.dtype).replace("torch.", ""),
            }
        )
    if len(inherited_mapping) != expected_inherited:
        raise FullVisualError("inherited mapping count differs")
    actual_attention = str(getattr(vision_tower.vision_model.config, "_attn_implementation", ""))
    if actual_attention != "eager":
        raise FullVisualError(
            f"standalone vision attention implementation is {actual_attention!r}, expected eager"
        )
    geometry = __import__(
        "owac.baselines.raw_stem_siglip", fromlist=["VisionGeometry"]
    ).VisionGeometry.from_modules(vision_tower, projector)
    patch_stride = geometry.packed_patch_stride(str(student_cfg.get("input_mode", "rawvla_rgb3")))
    stem = LocalConvStem(
        int(arch["num_channels"]),
        int(arch["hidden_size"]),
        patch_stride,
        conv_channels=int(student_cfg.get("stem_hidden", 64)),
        seed=int(student_cfg.get("stem_seed", 101)),
        dtype=target_dtype,
        device="cpu",
    )
    vision_tower.vision_model.embeddings.patch_embedding = stem
    # Legacy checkpoint/evidence identity fields are intentionally not created
    # on the active path.  Tensor values and shapes were compared directly above.
    prior_audit_path = Path(str(assets.get("load_audit") or ""))
    prior_audit = _load_json(prior_audit_path) if prior_audit_path.is_file() else None
    audit = InitializationAudit(
        source_tensors=(),
        student_tensors=(),
        stem_tensors=tuple(name for name, _ in stem.named_parameters()),
        checkpoint_sha256="",
        load_audit_sha256="",
        evidence_level="standalone_visual_subset",
    )
    base = RawStemSiglipStudent(
        vision_tower=vision_tower,
        projector=projector,
        input_mode=str(student_cfg.get("input_mode", "rawvla_rgb3")),
        evidence_level="standalone_visual_subset",
        checkpoint=None,
        load_audit=None,
        initialization_audit=audit,
        unfreeze_input_blocks=0,
        allow_input_block_unfreeze=False,
    )
    student = TemporalRawStemStudent(base)
    report = {
        "schema": "owac.raw-full-visual.student-build.v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": None,
        "source_visual_keys": len(inherited_mapping),
        "source_header_keys": len(header),
        "inherited_tensor_count": len(inherited_mapping),
        "student_fingerprint_count": 0,
        "inherited_values_equal_after_bf16_fp32_cast": len(inherited_mapping),
        "inherited_mapping": inherited_mapping,
        "prior_load_audit": {
            "path": str(prior_audit_path),
            "sha256": None,
            "linked": prior_audit is not None,
            "loaded_keys": None
            if not isinstance(prior_audit, Mapping)
            else prior_audit.get("loaded_keys"),
            "model_keys": None
            if not isinstance(prior_audit, Mapping)
            else prior_audit.get("model_keys"),
            "used_as_full_policy_acceptance": False,
        },
        "audit": {
            "evidence_level": "standalone_visual_subset",
            "acceptance": "structural_tensor_value_equality_no_hash",
            "source_fingerprints": 0,
            "student_fingerprints": 0,
            "empty_student_audit": True,
            "backend_loaded": False,
            "teacher_loaded": False,
            "patch_stem_replaced_fresh_seed": int(student_cfg.get("stem_seed", 101)),
        },
        "forward_geometry": {
            "module": (
                "starVLA.model.modules.vlm.openpi_transformers.siglip."
                "modeling_siglip.SiglipVisionModel"
            ),
            "image_size": int(arch["image_size"]),
            "patch_size": int(arch["patch_size"]),
            "hidden_size": int(arch["hidden_size"]),
            "num_hidden_layers": int(arch["num_hidden_layers"]),
            "num_patches": int(geometry.num_patches),
            "input_mode": str(student_cfg.get("input_mode", "rawvla_rgb3")),
            "patch_stride": int(patch_stride),
            "raw_resize": str(student_cfg.get("raw_resize", "")),
            "physical_position_backmap": str(student_cfg.get("physical_position_backmap", "")),
            "attention_implementation": actual_attention,
            "hidden_act": str(getattr(vision_tower.vision_model.config, "hidden_act", "")),
            "layer_norm_eps": float(geometry.post_layernorm_eps),
            "grid_height": int(geometry.grid_height),
            "grid_width": int(geometry.grid_width),
            "num_positions": int(geometry.num_positions),
            "encoder_layers": int(geometry.encoder_layers),
            "projector_in": int(geometry.projector_in),
            "projector_out": int(geometry.projector_out),
            "projected_token_order": "camera_major_then_patch_row_major",
            "projected_token_scale": "linear_projector_only_no_extra_normalization",
            "position_table_shared_across_cameras": True,
        },
        "stem_seed": int(student_cfg.get("stem_seed", 101)),
        "stem_fresh": True,
        "gate_fresh": True,
        "parameters_total": int(sum(p.numel() for p in student.parameters())),
        "parameter_count_expected": EXPECTED_FULL_VISUAL_PARAMETERS,
        "dtype": str(target_dtype).replace("torch.", ""),
        "backend_loaded": False,
        "teacher_loaded": False,
        "audit_sha256": None,
        "checkpoint_identity": {"path": str(checkpoint), "sha256": None},
    }
    return student.to(device), report


def configure_full_visual_scope(student: Any) -> dict[str, Any]:
    """Mark exactly the full visual visual head trainable; gate included."""
    named = dict(student.named_parameters())
    selection = select_full_visual_trainable(named)
    if not selection["matches_expected"]:
        raise FullVisualError(
            "full visual parameter selection differs from 428,579,952 or has rejected names: "
            f"{selection['categories']} rejected={selection['rejected'][:8]}"
        )
    for name, parameter in named.items():
        parameter.requires_grad_(str(name) in set(selection["selected"]))
    student._scope = FULL_VISUAL_SCOPE
    if hasattr(student, "base"):
        student.base._scope = FULL_VISUAL_SCOPE
    active = [(name, parameter) for name, parameter in named.items() if parameter.requires_grad]
    if sum(parameter.numel() for _, parameter in active) != EXPECTED_FULL_VISUAL_PARAMETERS:
        raise FullVisualError("active full visual parameter count differs after requires_grad")
    for category, count in selection["categories"].items():
        if count <= 0:
            raise FullVisualError(f"full visual category {category} is empty")
    return selection


def trainable_student_parameters(student: Any) -> list[Any]:
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    total = sum(parameter.numel() for parameter in parameters)
    if total != EXPECTED_FULL_VISUAL_PARAMETERS:
        raise FullVisualError(
            f"trainable parameter count {total} differs from {EXPECTED_FULL_VISUAL_PARAMETERS}"
        )
    return parameters


# --------------------------------------------------------------------------- #
# DDP adapter and forward reuse
# --------------------------------------------------------------------------- #
def set_full_visual_train_mode(student: Any, mode: bool = True) -> None:
    """Explicit full-visual semantics: vision/projector follow train mode.

    The inherited freeze-stage ``TemporalRawStemStudent.train`` forces the SigLIP
    tower and projector to eval regardless of requires_grad.  That is not the new
    full-visual stage semantics, so set them explicitly after any inherited call.
    """
    student.train(bool(mode))
    if hasattr(student, "base"):
        student.base.vision_tower.train(bool(mode))
        student.base.projector.train(bool(mode))


class DDPMathAdapter:
    """Expose temporal student methods while routing the final forward via DDP."""

    def __init__(self, ddp: Any) -> None:
        self.ddp = ddp
        self.module = ddp.module

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)

    def update_state(self, packet: Any, state: Any) -> Any:
        return self.module.update_state(packet, state)

    def detach_state(self, state: Any) -> Any:
        return self.module.detach_state(state)

    def __call__(self, packet: Any, state: Any = None) -> Any:
        # This is the actual invoked final endpoint path; DDP hooks fire here.
        return self.ddp(packet, state)

    def forward_batched(self, samples: Any) -> Any:
        from owac.training.raw_full_visual_batched import BatchedTemporalRequest

        # One DDP module call per batch; the ordinary reducer fires on backward.
        return self.ddp(BatchedTemporalRequest(tuple(samples)))


def _checkpoint_wrap_vision(student: Any) -> int:
    """Per-SigLIP-block activation checkpointing on the actual forward path."""
    import torch

    layers = student.base.vision_tower.vision_model.encoder.layers
    count = 0
    for layer in layers:
        original = layer.forward
        if getattr(layer, "_owac_full_visual_checkpoint", False):
            continue

        def wrapped(*args: Any, _original: Any = original, _layer: Any = layer, **kwargs: Any):
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return torch.utils.checkpoint.checkpoint(
                _original, *args, use_reentrant=False, **kwargs
            )

        layer.forward = wrapped
        layer._owac_full_visual_checkpoint = True
        count += 1
    return count


# --------------------------------------------------------------------------- #
# Checkpoint retention (split model/optimizer files)
# --------------------------------------------------------------------------- #
def plan_checkpoint_retention(
    unique_bytes: int,
    *,
    cap_bytes: int,
    reserve_bytes: int,
) -> dict[str, Any]:
    """Pure capacity plan; latest+best are the maximum retained full/state set."""
    if unique_bytes < 0 or cap_bytes <= 0 or reserve_bytes < 0:
        raise FullVisualError("checkpoint retention arguments are invalid")
    projected = int(unique_bytes) + int(reserve_bytes)
    return {
        "unique_bytes": int(unique_bytes),
        "reserve_bytes": int(reserve_bytes),
        "cap_bytes": int(cap_bytes),
        "projected_bytes": projected,
        "fits": projected <= int(cap_bytes),
        "policy": "step0_model_only_plus_latest_full_plus_best_model",
        "max_full_checkpoints": 2,
    }


def assert_checkpoint_budget(root: Path, *, reserve_bytes: int, cap_bytes: int) -> int:
    used = owned_run_bytes(root)
    plan = plan_checkpoint_retention(used, cap_bytes=cap_bytes, reserve_bytes=reserve_bytes)
    if not plan["fits"]:
        raise FullVisualError(
            f"checkpoint output would exceed cap: {plan['projected_bytes']} > {cap_bytes}"
        )
    return used


# --------------------------------------------------------------------------- #
# Distributed runtime helpers
# --------------------------------------------------------------------------- #
def _config_digest(config: Mapping[str, Any]) -> str:
    """Return explicit structural config text; this is not a hash."""
    value = copy.deepcopy(dict(config))
    value.pop("authorization", None)
    remote = value.get("remote")
    if isinstance(remote, dict):
        remote.pop("output_host", None)
        remote.pop("host", None)
        remote.pop("ssh_port", None)
        remote.pop("output_root", None)
        remote.pop("path_mapping", None)
    return _canonical_json(value)


def _init_distributed(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise FullVisualError("this command must be launched by torchrun (RANK/WORLD_SIZE absent)")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    expected = int((config.get("remote") or {}).get("world_size", DEFAULT_WORLD_SIZE))
    if world_size != expected or world_size != 2:
        raise FullVisualError(
            "this stage requires exactly two ranks, got "
            f"world_size={world_size} expected={expected}"
        )
    if not torch.cuda.is_available():
        raise FullVisualError("CUDA is not available; this runtime is GPU-only")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise FullVisualError("LOCAL_RANK is outside available CUDA devices")
    backend = str((config.get("remote") or {}).get("backend", "nccl"))
    dist.init_process_group(backend=backend, timeout=__import__("datetime").timedelta(seconds=300))
    torch.cuda.set_device(local_rank)
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": f"cuda:{local_rank}",
        "backend": backend,
    }


def _barrier(timeout: float = 180.0) -> None:
    import datetime as _datetime

    import torch.distributed as dist

    dist.barrier(timeout=_datetime.timedelta(seconds=float(timeout)))


def _all_reduce_int(value: int) -> int:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([int(value)], dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_max_float(value: float) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _all_reduce_sum_float(value: float) -> float:
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([float(value)], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _gather_rank_processes(distributed: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch.distributed as dist

    local = {
        "rank": int(distributed["rank"]),
        "local_rank": int(distributed["local_rank"]),
        "pid": os.getpid(),
        "pid_start_ticks": _proc_start_ticks(os.getpid()),
        "pgid": os.getpgid(0),
    }
    gathered = [None] * int(distributed["world_size"])
    dist.all_gather_object(gathered, local)
    return [dict(item) for item in gathered if isinstance(item, Mapping)]


def _gather_gpu_stats(distributed: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch
    import torch.distributed as dist

    index = int(distributed["local_rank"])
    local = {
        "rank": int(distributed["rank"]),
        "local_rank": index,
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }
    gathered = [None] * int(distributed["world_size"])
    dist.all_gather_object(gathered, local)
    return [dict(item) for item in gathered if isinstance(item, Mapping)]


def _cpuize(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {str(key): _cpuize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpuize(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return {
            "__torch_state__": True,
            "dtype": str(value.dtype).replace("torch.", ""),
            "shape": list(value.shape),
            "data": value.detach().cpu().reshape(-1).tolist(),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _state_tensor(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping) and value.get("__torch_state__") is True:
        dtype = getattr(torch, str(value["dtype"]))
        tensor = torch.tensor(value["data"], dtype=dtype)
        return tensor.reshape(tuple(int(item) for item in value["shape"]))
    raise FullVisualError("RNG state tensor encoding differs")


def _restore_rng_state(value: Mapping[str, Any], device: str) -> None:
    import random

    import numpy as np
    import torch

    if "python" in value:

        def _tuple_tree(item: Any) -> Any:
            if isinstance(item, list):
                return tuple(_tuple_tree(child) for child in item)
            return item

        random.setstate(_tuple_tree(value["python"]))
    if "numpy" in value:
        state = value["numpy"]
        keys = state["keys"]
        np.random.set_state(
            (
                str(state["kind"]),
                _state_tensor(keys).detach().cpu().numpy().astype(np.uint32),
                int(state["pos"]),
                int(state["has_gauss"]),
                float(state["cached_gaussian"]),
            )
        )
    if "torch" in value:
        torch.set_rng_state(_state_tensor(value["torch"]).detach().cpu())
    cuda = value.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        index = int(str(device).split(":")[-1])
        if isinstance(cuda, Mapping):
            if cuda.get("schema") != RANK_LOCAL_CUDA_RNG_SCHEMA:
                raise FullVisualError("rank-local CUDA RNG schema differs")
            saved_index = int(cuda.get("device_index", -1))
            if saved_index != index:
                raise FullVisualError(
                    f"rank-local CUDA RNG device {saved_index} != target device {index}"
                )
            cuda_state = _state_tensor(cuda["state"])
        elif isinstance(cuda, (list, tuple)):
            # Backward compatibility only for the unambiguous old rank-local
            # list (length one).  Longer lists are legacy global lists and may
            # be indexed only when this rank is present.
            if len(cuda) == 1:
                cuda_state = _state_tensor(cuda[0])
            elif 0 <= index < len(cuda):
                cuda_state = _state_tensor(cuda[index])
            else:
                raise FullVisualError(
                    f"legacy CUDA RNG list has no unambiguous state for device {index}"
                )
        else:
            raise FullVisualError("CUDA RNG state has unsupported shape")
        torch.cuda.set_rng_state(cuda_state.detach().cpu(), device=index)


def _capture_rng_state(device: str) -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    index = int(str(device).split(":")[-1])
    return {
        "python": random.getstate(),
        "numpy": {
            "kind": str(numpy_state[0]),
            "keys": torch.from_numpy(np.asarray(numpy_state[1], dtype=np.uint32).copy()),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": {
            "schema": RANK_LOCAL_CUDA_RNG_SCHEMA,
            "device_index": index,
            "state": torch.cuda.get_rng_state(index).detach().cpu(),
        },
    }


@dataclass
class FullVisualRuntime:
    config: dict[str, Any]
    mapped_config: dict[str, Any]
    mapping_report: dict[str, Any]
    run_root: Path
    cache: ImmutableTargetCache
    data: Any
    dataset: Any
    student: Any
    ddp: Any
    adapter: Any
    trainer: Any
    optimizer: Any
    scheduler: Any
    sampler: Any
    candidates: list[tuple[str, int]]
    report: dict[str, Any]
    split: dict[str, tuple[str, ...]]
    budget: FullVisualBudget
    distributed: dict[str, Any]
    seed: int
    preflight_receipt: dict[str, Any] | None = None
    start_new_update: int = 0
    start_anchor_presentations: int = 0
    budget_gpu_wall_seconds: float = 0.0
    checkpoint_meta: dict[str, Any] | None = None
    cache_report: dict[str, Any] = field(default_factory=dict)
    finalization_reserve_seconds: float = 600.0
    rank_processes: list[dict[str, Any]] = field(default_factory=list)
    gpu_by_rank: list[dict[str, Any]] = field(default_factory=list)
    data_contract: dict[str, Any] = field(default_factory=dict)
    teacher_prepare_receipt: dict[str, Any] | None = None
    performance_options: PerformanceOptions = field(default_factory=PerformanceOptions)
    fastdata_report: dict[str, Any] = field(default_factory=dict)


def _teacher_receipt(config: Mapping[str, Any], mapper: PathMapper) -> dict[str, Any] | None:
    remote = dict(config.get("remote") or {})
    required = bool(remote.get("require_teacher_preparation_receipt", False))
    raw_path = remote.get("teacher_preparation_receipt")
    if not raw_path:
        if required:
            raise FullVisualError("shared stage requires a teacher preparation receipt path")
        return None
    resolution = mapper.resolve(str(raw_path), must_exist=required)
    path = Path(resolution.physical)
    if not path.is_file():
        if required:
            raise FullVisualError(f"teacher preparation receipt is absent: {path}")
        return None
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise FullVisualError("teacher preparation receipt is not an object")
    if payload.get("schema") != "owac.shared-rawvla.teacher-preparation.v1":
        raise FullVisualError("teacher preparation receipt schema differs")
    if payload.get("protocol") != SHARED_PROTOCOL:
        raise FullVisualError("teacher preparation receipt protocol differs")
    if payload.get("test_targets_generated") is not False:
        raise FullVisualError("teacher preparation receipt generated test targets")
    if payload.get("actual_source_hdf5_sha256") is not None:
        raise FullVisualError("teacher receipt claims an unverified actual HDF5 SHA")
    if payload.get("actual_source_hdf5_sha256_available") is not False:
        raise FullVisualError("teacher receipt incorrectly claims actual HDF5 availability")
    if payload.get("paired_rgb_used") is not False:
        raise FullVisualError("teacher receipt claims paired RGB was used")
    key = str(payload.get("cache_identity_key") or "")
    if not key or key in {".", ".."} or Path(key).name != key or "/" in key or "\\" in key:
        raise FullVisualError(
            "teacher preparation cache identity must be a nonempty safe directory component"
        )
    wall = float(payload.get("gpu_wall_sum_seconds", -1))
    if not math.isfinite(wall) or wall < 0:
        raise FullVisualError("teacher preparation GPU wall is invalid")
    return {
        "path": str(path),
        "sha256": None,
        "mapped": bool(resolution.mapped),
        "payload": dict(payload),
    }


def validate_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    config_digest: str,
    cache_identity_key: str,
    teacher_preparation_gpu_wall_seconds: float,
) -> dict[str, Any]:
    """Accept old phase-only receipts while carrying prep wall, never resetting it."""
    if not isinstance(receipt, Mapping) or receipt.get("schema") != PREFLIGHT_SCHEMA:
        raise FullVisualError("preflight receipt schema differs")
    # Legacy config/cache digest fields are compatibility metadata only.
    prep_baseline = float(teacher_preparation_gpu_wall_seconds)
    if not math.isfinite(prep_baseline) or prep_baseline < 0:
        raise FullVisualError("teacher preparation wall baseline is invalid")
    phase_raw = receipt.get("preflight_phase_gpu_wall_seconds")
    if phase_raw is None:
        # Old preflight receipts emitted only this phase elapsed time.  It is a
        # phase value, not a cumulative budget, so add the already-loaded prep
        # charge exactly once.
        phase_raw = receipt.get("gpu_wall_sum_seconds")
    phase = float(phase_raw if phase_raw is not None else -1)
    if not math.isfinite(phase) or phase < 0:
        raise FullVisualError("preflight phase GPU wall is invalid")
    prep_claimed = receipt.get("teacher_preparation_gpu_wall_seconds")
    if prep_claimed is not None and abs(float(prep_claimed) - prep_baseline) > 1e-6:
        raise FullVisualError("preflight receipt teacher prep wall differs from loaded receipt")
    cumulative = receipt.get("cumulative_gpu_wall_seconds")
    if cumulative is None:
        cumulative = prep_baseline + phase
    cumulative = float(cumulative)
    if not math.isfinite(cumulative) or cumulative < 0:
        raise FullVisualError("preflight cumulative GPU wall is invalid")
    if cumulative + 1e-6 < prep_baseline + phase:
        raise FullVisualError("preflight cumulative GPU wall drops phase or preparation charge")
    return {
        "phase_gpu_wall_seconds": phase,
        "teacher_preparation_gpu_wall_seconds": prep_baseline,
        "cumulative_gpu_wall_seconds": cumulative,
        "gpu_wall_sum_seconds": cumulative,
    }


def _build_runtime(
    config_path: str | Path,
    *,
    run_root: str | Path | None,
    distributed: Mapping[str, Any],
    config_override: Mapping[str, Any] | None = None,
    resume_checkpoint: str | Path | None = None,
    preflight_receipt: str | Path | None = None,
    performance_config: str | Path | None = None,
) -> FullVisualRuntime:
    import torch
    from torch import nn

    from owac.raw.teacher_data import TeacherData, load_source_manifest
    from owac.raw.temporal_distillation import TemporalPacketDataset
    from owac.training.raw_temporal_distillation import (
        TemporalAnchorSampler,
        TemporalDistillationTrainer,
        TemporalTrainingConfig,
    )

    config = dict(config_override) if config_override is not None else dict(_load_json(config_path))
    performance_options = load_performance_options(performance_config)
    mapper = PathMapper((config.get("remote") or {}).get("path_mapping") or ())
    mapped_config, mapping_report = _map_config_paths(config, mapper)
    remote = dict(config.get("remote") or {})
    root_value = run_root or remote.get("output_root") or mapped_config["assets"].get("output_root")
    if not isinstance(root_value, str) or not root_value:
        raise FullVisualError("run root is missing")
    root = Path(root_value)
    root.mkdir(parents=True, exist_ok=True)
    if str(remote.get("path_alias") or "") and not Path(str(remote["path_alias"])).exists():
        raise FullVisualError(
            "the remote logical-path alias is absent; create the approved symlink before running"
        )
    budget = FullVisualBudget.from_config(config)
    data_contract = _data_contract(config)
    receipt_info = _teacher_receipt(config, mapper)
    if data_contract["protocol"] == SHARED_PROTOCOL and receipt_info is None:
        raise FullVisualError("shared stage requires teacher preparation receipt")
    assets_cfg = dict(mapped_config.get("assets") or {})
    from owac.raw.temporal_distillation import TemporalAssetSet

    assets = TemporalAssetSet.from_config(mapped_config, repo_root=REPO_ROOT)
    if data_contract["protocol"] == SHARED_PROTOCOL:
        # The historical asset validator hard-codes the old 50-source manifest
        # hash.  Shared preparation validates teacher assets itself and binds
        # the new manifest/cache through the signed receipt.
        if receipt_info is None:
            raise FullVisualError("shared stage requires teacher preparation receipt")
        # Legacy receipt/source hash fields are compatibility metadata only.
    # Historical asset digest fields are compatibility metadata only.
    cache_cfg = dict(config.get("cache") or {})
    receipt_key = (
        str(receipt_info["payload"]["cache_identity_key"]) if receipt_info is not None else ""
    )
    configured_key = str(cache_cfg.get("identity_key") or "")
    expected_cache_key = receipt_key or configured_key
    cache = ImmutableTargetCache(assets_cfg["cache_root"], expected_identity_key=expected_cache_key)
    if data_contract["protocol"] == SHARED_PROTOCOL:
        if receipt_info is None:
            raise FullVisualError("shared stage requires teacher preparation receipt")
        # Cache key remains an opaque compatibility path; no hash is computed or
        # compared.  Row shape/finite/source/endpoint checks remain in the reader.
        if (
            Path(assets_cfg["cache_root"]).resolve()
            != Path(receipt_info["payload"].get("cache_root", "")).resolve()
        ):
            raise FullVisualError("cache root differs from teacher preparation receipt")
    if data_contract["protocol"] == SHARED_PROTOCOL:
        from owac.raw.shared_rawvla_distillation import load_shared_manifest

        manifest = load_shared_manifest(assets.sources_path, repo_root=REPO_ROOT)
    else:
        manifest = load_source_manifest(
            assets.sources_path,
            repo_root=REPO_ROOT,
            strict=True,
            expected_splits=data_contract["expected_splits"],
        )
    if len(manifest.sources) != data_contract["expected_source_count"]:
        raise FullVisualError(
            f"source count {len(manifest.sources)} != {data_contract['expected_source_count']}"
        )
    data = TeacherData(
        manifest,
        repo_root=REPO_ROOT,
        cache_entries=int(performance_options.episode_cache_entries),
    )
    fastdata_report: dict[str, Any] = {}
    if performance_options.raw_preload:
        fastdata_report["raw_preload"] = preload_raw_episodes(data, manifest)
    split = deterministic_source_split(manifest, expected_splits=data_contract["expected_splits"])
    student, student_report = build_full_visual_student(
        mapped_config, device=distributed["device"], dtype=torch.float32
    )
    scope_report = configure_full_visual_scope(student)
    set_full_visual_train_mode(student, True)
    trainable = trainable_student_parameters(student)
    checkpointing_blocks = (
        _checkpoint_wrap_vision(student)
        if bool((config.get("training") or {}).get("activation_checkpointing", True))
        else 0
    )
    student.to(distributed["device"])
    for parameter in student.parameters():
        if not parameter.requires_grad:
            parameter.requires_grad_(False)
    ddp = nn.parallel.DistributedDataParallel(
        student,
        device_ids=[distributed["local_rank"]],
        output_device=distributed["local_rank"],
        broadcast_buffers=False,
        find_unused_parameters=bool(
            (config.get("training") or {}).get("find_unused_parameters", False)
        ),
        gradient_as_bucket_view=True,
    )
    adapter = DDPMathAdapter(ddp)
    settings_mapping = dict(config.get("training") or {})
    settings_mapping["microbatch"] = 1
    settings_mapping["activation_memory_mode"] = None
    settings = TemporalTrainingConfig.from_mapping(settings_mapping)
    trainer = TemporalDistillationTrainer(
        student=adapter,
        bridge=None,
        config=settings,
        arm="C",
        teacher=None,
        run_identity={},
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings_mapping.get("learning_rate", 1e-5)),
        weight_decay=float(settings_mapping.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0, (step + 1) / max(1, int(settings_mapping.get("warmup_updates", 100)))
        ),
    )
    preload_ids = [
        str(source.source_demo_id)
        for split in ("train", "validation")
        for source in manifest.sources_for_split(split)
    ]
    cache_report = cache.preload(preload_ids)
    expected_cache_rows = int(data_contract["expected_cache_rows"])
    if int(cache_report["rows"]) != expected_cache_rows:
        raise FullVisualError(
            f"train+validation cache row count {cache_report['rows']} != "
            f"frozen {expected_cache_rows}"
        )
    if int(distributed["rank"]) == 0:
        _atomic_json(root / "cache-preload.json", cache_report)
        print(
            json.dumps(
                {
                    "event": "teacher_cache_preload",
                    "rows": cache_report["rows"],
                    "file_bytes": cache_report["file_bytes"],
                    "load_seconds": cache_report["load_seconds"],
                    "rss_after_bytes": cache_report["rss_after_bytes"],
                    "scope": cache_report["scope"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    candidates: list[tuple[str, int]] = []
    for source in manifest.sources_for_split("train"):
        source_id = str(source.source_demo_id)
        for endpoint in range(25, int(source.num_frames), 5):
            if not cache.exists(source_id, endpoint):
                raise FullVisualError(f"training cache row is missing: {source_id}@{endpoint}")
            candidates.append((source_id, endpoint))
    for source in manifest.sources_for_split("validation"):
        source_id = str(source.source_demo_id)
        for endpoint in range(0, int(source.num_frames), 5):
            if not cache.exists(source_id, endpoint):
                raise FullVisualError(f"validation cache row is missing: {source_id}@{endpoint}")
    expected_train_anchors = int(data_contract["expected_train_anchors"])
    if len(candidates) != expected_train_anchors:
        raise FullVisualError(
            f"expected {expected_train_anchors} training candidate anchors, got {len(candidates)}"
        )
    validation_rows = sum(
        len(range(0, int(source.num_frames), 5))
        for source in manifest.sources_for_split("validation")
    )
    if validation_rows != int(data_contract["expected_validation_windows"]):
        raise FullVisualError(
            "validation windows "
            f"{validation_rows} != {data_contract['expected_validation_windows']}"
        )
    sampler = TemporalAnchorSampler(
        candidates,
        seed=int(settings_mapping.get("seed", 101)),
        batch_size=int(settings_mapping.get("effective_batch", 16)),
    )
    if performance_options.packet_cache_max_bytes > 0:
        dataset = CachedPacketDataset(
            data=data,
            cache=cache,
            student=student,
            arm="C",
            packet_cache_max_bytes=int(performance_options.packet_cache_max_bytes),
        )
    else:
        dataset = TemporalPacketDataset(data=data, cache=cache, student=student, arm="C")
    runtime = FullVisualRuntime(
        config=config,
        mapped_config=mapped_config,
        mapping_report=mapping_report,
        run_root=root,
        cache=cache,
        data=data,
        dataset=dataset,
        student=student,
        ddp=ddp,
        adapter=adapter,
        trainer=trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        candidates=candidates,
        report={
            **student_report,
            "scope": scope_report,
            "checkpointing_blocks": checkpointing_blocks,
        },
        split=split,
        budget=budget,
        distributed=dict(distributed),
        seed=int(settings_mapping.get("seed", 101)),
        cache_report=cache_report,
        data_contract=data_contract,
        teacher_prepare_receipt=receipt_info,
        performance_options=performance_options,
        fastdata_report=fastdata_report,
        finalization_reserve_seconds=float(
            (config.get("training") or {}).get("finalization_reserve_seconds", 600.0)
        ),
    )
    if receipt_info is not None and runtime.checkpoint_meta is None:
        runtime.budget_gpu_wall_seconds = float(receipt_info["payload"]["gpu_wall_sum_seconds"])
        runtime.budget.assert_gpu_usage(runtime.budget_gpu_wall_seconds)
    if resume_checkpoint is not None:
        _load_full_visual_checkpoint(runtime, Path(resume_checkpoint))
    if preflight_receipt is not None and runtime.checkpoint_meta is None:
        receipt = _load_json(preflight_receipt)
        carried = validate_preflight_receipt(
            receipt,
            config_digest=_config_digest(config),
            cache_identity_key=expected_cache_key,
            teacher_preparation_gpu_wall_seconds=runtime.budget_gpu_wall_seconds,
        )
        runtime.budget_gpu_wall_seconds = carried["cumulative_gpu_wall_seconds"]
        runtime.finalization_reserve_seconds = float(
            receipt.get("finalization_reserve_seconds", runtime.finalization_reserve_seconds)
        )
        runtime.preflight_receipt = {
            "path": str(Path(preflight_receipt)),
            "sha256": None,
            "preflight_phase_gpu_wall_seconds": carried["phase_gpu_wall_seconds"],
            "teacher_preparation_gpu_wall_seconds": carried["teacher_preparation_gpu_wall_seconds"],
            "cumulative_gpu_wall_seconds": carried["cumulative_gpu_wall_seconds"],
            "gpu_wall_sum_seconds": carried["gpu_wall_sum_seconds"],
        }
    return runtime


# --------------------------------------------------------------------------- #
# Checkpoint/status (rank 0)
# --------------------------------------------------------------------------- #
def _checkpoint_root(run_root: Path) -> Path:
    path = run_root / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _update_latest_pointer(run_root: Path, checkpoint: Path, meta: Mapping[str, Any]) -> None:
    _atomic_json(
        run_root / "latest.json",
        {
            "schema": "owac.raw-full-visual.latest.v1",
            "path": str(checkpoint.relative_to(run_root)),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "model_sha256": meta.get("model_sha256"),
            "optimizer_sha256": meta.get("optimizer_sha256"),
            "new_global_updates": meta.get("new_global_updates"),
            "kind": meta.get("kind"),
            "updated_at": _now_iso(),
        },
    )


def _best_pointer(run_root: Path, checkpoint: Path, meta: Mapping[str, Any]) -> None:
    best = run_root / "best"
    best.mkdir(parents=True, exist_ok=True)
    source_model = checkpoint / "model.pt"
    target_model = best / "model.pt"
    if target_model.exists():
        target_model.unlink()
    os.link(source_model, target_model)
    _fsync_dir(best)
    _atomic_json(
        best / "meta.json",
        {
            "schema": "owac.raw-full-visual.best.v1",
            "source_checkpoint": str(checkpoint.relative_to(run_root)),
            "checkpoint_sha256": meta.get("checkpoint_sha256"),
            "model_sha256": meta.get("model_sha256"),
            "new_global_updates": meta.get("new_global_updates"),
            "validation": meta.get("validation"),
            "updated_at": _now_iso(),
        },
    )


def _prune_old_checkpoints(run_root: Path, *, keep: set[Path]) -> None:
    for path in sorted((run_root / "checkpoints").iterdir()):
        if not path.is_dir() or path in keep:
            continue
        if path.name.startswith("."):
            continue
        import shutil

        shutil.rmtree(path)


def prepare_checkpoint_room(runtime: FullVisualRuntime, *, reserve_bytes: int) -> int:
    """Prune only owned recovery artifacts before a potentially 5.15GB save.

    Step0 is model-only and may be removed when it is not the best source; the
    latest full checkpoint and the best model hardlink are retained.  Never
    touches the old local run or replicated immutable assets.
    """
    import shutil

    cap = int(runtime.budget.remote_output_cap_bytes)
    root = runtime.run_root
    used = owned_run_bytes(root)
    if used + int(reserve_bytes) <= cap:
        return used
    checkpoints = root / "checkpoints"
    step0 = checkpoints / "step0"
    best_source = None
    best_meta_path = root / "best" / "meta.json"
    if best_meta_path.is_file():
        best_meta = _load_json(best_meta_path)
        if isinstance(best_meta, Mapping):
            best_source = str(best_meta.get("source_checkpoint") or "") or None
    if step0.is_dir() and best_source != "checkpoints/step0":
        shutil.rmtree(step0)
    latest_path = None
    latest_file = root / "latest.json"
    if latest_file.is_file():
        latest = _load_json(latest_file)
        if isinstance(latest, Mapping):
            latest_path = str(latest.get("path") or "") or None
    used = owned_run_bytes(root)
    if checkpoints.is_dir():
        for path in sorted(checkpoints.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            relative = f"checkpoints/{path.name}"
            if relative in {latest_path, best_source}:
                continue
            shutil.rmtree(path)
    used = owned_run_bytes(root)
    if used + int(reserve_bytes) > cap:
        raise FullVisualError(
            f"checkpoint would exceed remaining output cap after pruning: "
            f"{used + int(reserve_bytes)} > {cap}"
        )
    return used


def _save_full_visual_checkpoint(
    runtime: FullVisualRuntime,
    *,
    kind: str,
    include_optimizer: bool,
    selection: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    gpu_wall_sum_seconds: float,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    rank = int(runtime.distributed["rank"])
    world_size = int(runtime.distributed["world_size"])
    root = runtime.run_root
    checkpoints = _checkpoint_root(root)
    name = checkpoint_name or f"checkpoint_new_{int(runtime.start_new_update):06d}"
    final_dir = checkpoints / name
    if final_dir.exists():
        raise FullVisualError(f"checkpoint directory already exists: {final_dir}")
    reserve = int(
        (runtime.config.get("training") or {}).get("checkpoint_reserve_bytes", 5_200_000_000)
    )
    used = prepare_checkpoint_room(runtime, reserve_bytes=reserve)
    rng_local = _capture_rng_state(runtime.distributed["device"])
    # Object collectives must carry primitives only.  raw torch RNG tensors can
    # hit torch.storage shims in mixed Torch/CUDA environments; serialize first
    # with the same JSON state encoding used by checkpoints/reload.
    rng_objects = [None] * world_size
    dist.all_gather_object(rng_objects, _jsonable(_cpuize(rng_local)))
    sampler_objects = [None] * world_size
    dist.all_gather_object(sampler_objects, _jsonable(runtime.sampler.state()))
    result: dict[str, Any] = {}
    if rank == 0:
        temp = checkpoints / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        temp.mkdir(parents=True)
        try:
            model_path = temp / "model.pt"
            model_state = {
                key: value.detach().cpu() for key, value in runtime.ddp.module.state_dict().items()
            }
            torch.save(model_state, model_path)
            _fsync_file(model_path)
            model_sha = None
            optimizer_path = None
            optimizer_sha = None
            if include_optimizer:
                optimizer_path = temp / "optimizer.pt"
                torch.save(runtime.optimizer.state_dict(), optimizer_path)
                _fsync_file(optimizer_path)
                optimizer_sha = None
                scheduler_path = temp / "scheduler.pt"
                torch.save(runtime.scheduler.state_dict(), scheduler_path)
                _fsync_file(scheduler_path)
            meta = {
                "schema": CHECKPOINT_SCHEMA,
                "kind": str(kind),
                "arm": "C",
                "scope": FULL_VISUAL_SCOPE,
                "world_size": world_size,
                "model_sha256": model_sha,
                "optimizer_sha256": optimizer_sha,
                "checkpoint_sha256": model_sha,
                "model_path": "model.pt",
                "optimizer_path": None if optimizer_path is None else "optimizer.pt",
                "new_global_updates": int(runtime.start_new_update),
                "new_anchor_presentations": int(runtime.start_anchor_presentations),
                "budget_update_equivalents": int(
                    PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
                ),
                "budget_anchor_equivalents": int(
                    PARENT_ANCHORS_EQUIVALENTS_SPENT + runtime.start_anchor_presentations
                ),
                "remote_gpu_wall_used_seconds": float(gpu_wall_sum_seconds),
                "remote_output_bytes": int(used + reserve),
                "preflight_receipt": runtime.preflight_receipt,
                "teacher_preparation_receipt": runtime.teacher_prepare_receipt,
                "data_contract": runtime.data_contract,
                "performance": runtime.performance_options.as_dict(),
                "fastdata": runtime.fastdata_report,
                "finalization_reserve_seconds": runtime.finalization_reserve_seconds,
                "cache_report": runtime.cache_report,
                "sampler": _jsonable(runtime.sampler.state()),
                "sampler_by_rank": sampler_objects,
                "rng_by_rank": rng_objects,
                "selection": None if selection is None else dict(selection),
                "validation": None if validation is None else dict(validation),
                "config_digest": _config_digest(runtime.config),
                "run_root": str(root),
                "source": runtime.report.get("source_checkpoint"),
                "source_sha256": None,
                "stem_fresh_seed": runtime.report.get("stem_seed"),
                "parent_student_checkpoint_used": False,
                "teacher_backend_optimizer_members": False,
                "created_at": _now_iso(),
            }
            _atomic_json(temp / "meta.json", meta)
            os.replace(temp, final_dir)
            _fsync_dir(checkpoints)
            result = dict(meta)
        except Exception:
            import shutil

            shutil.rmtree(temp, ignore_errors=True)
            raise
    dist.barrier(timeout=__import__("datetime").timedelta(seconds=300))
    meta_objects = [None] * world_size
    dist.all_gather_object(meta_objects, _jsonable(result) if rank == 0 else {})
    if rank != 0:
        result = dict(meta_objects[0] or {})
    if not result:
        raise FullVisualError("rank 0 did not publish checkpoint metadata")
    if rank == 0:
        _update_latest_pointer(root, final_dir, result)
        if validation is not None and validation.get("is_best"):
            _best_pointer(root, final_dir, result)
            best = root / "best" / "model.pt"
            result["best_model_path"] = str(best.relative_to(root))
        keep = {final_dir}
        best_dir = root / "best"
        if best_dir.is_dir():
            keep.add(best_dir)
        step0 = checkpoints / "step0"
        if step0.is_dir():
            keep.add(step0)
        _prune_old_checkpoints(root, keep=keep)
    return result


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def validate_checkpoint_metadata(
    meta: Mapping[str, Any],
    *,
    config_digest: str | None = None,
    source_sha256: str | None = None,
    world_size: int = 2,
) -> dict[str, Any]:
    """Pure strict checkpoint contract shared by save/reload tests."""
    if not isinstance(meta, Mapping) or meta.get("schema") != CHECKPOINT_SCHEMA:
        raise FullVisualError("checkpoint metadata schema differs")
    if meta.get("scope") != FULL_VISUAL_SCOPE:
        raise FullVisualError("checkpoint full-visual scope differs")
    kind = str(meta.get("kind") or "")
    if kind not in {"step0", "interval", "final"}:
        raise FullVisualError("checkpoint kind differs")
    if int(meta.get("world_size", 0)) != int(world_size):
        raise FullVisualError("checkpoint world size differs")
    # Legacy config/source/model digest fields are compatibility metadata only.
    if kind == "step0":
        if meta.get("optimizer_path") is not None or meta.get("optimizer_sha256") is not None:
            raise FullVisualError("step0 checkpoint must be model-only")
    else:
        if meta.get("optimizer_path") != "optimizer.pt":
            raise FullVisualError("full checkpoint must persist optimizer state")
    updates = int(meta.get("new_global_updates", -1))
    anchors = int(meta.get("new_anchor_presentations", -1))
    if updates < 0 or updates > 3116:
        raise FullVisualError("checkpoint new-update counter is outside the stage cap")
    if anchors != updates * EXPECTED_EFFECTIVE_BATCH:
        raise FullVisualError("checkpoint anchor counter differs from the global batch")
    if int(meta.get("budget_update_equivalents", -1)) != (
        PARENT_UPDATE_EQUIVALENTS_SPENT + updates
    ):
        raise FullVisualError("checkpoint budget-equivalent counter differs")
    if meta.get("parent_student_checkpoint_used", False):
        raise FullVisualError("checkpoint must not be bound to the old student checkpoint")
    rng = meta.get("rng_by_rank")
    if not isinstance(rng, list) or len(rng) != int(world_size):
        raise FullVisualError("checkpoint lacks per-rank RNG states")
    sampler = meta.get("sampler")
    if (
        not isinstance(sampler, Mapping)
        or int(sampler.get("batch_size", -1)) != EXPECTED_EFFECTIVE_BATCH
    ):
        raise FullVisualError("checkpoint sampler state differs")
    return {
        "kind": kind,
        "new_global_updates": updates,
        "new_anchor_presentations": anchors,
        "model_sha256": meta.get("model_sha256"),
        "optimizer_present": meta.get("optimizer_path") is not None,
    }


def _load_full_visual_checkpoint(runtime: FullVisualRuntime, checkpoint: Path) -> None:
    import torch

    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FullVisualError("resume checkpoint must be an explicit checkpoint directory")
    meta = _load_json(checkpoint / "meta.json")
    validate_checkpoint_metadata(
        meta,
        world_size=int(runtime.distributed["world_size"]),
    )
    model_path = checkpoint / "model.pt"
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    runtime.ddp.module.load_state_dict(state, strict=True)
    if meta.get("optimizer_path") is not None:
        optimizer_path = checkpoint / "optimizer.pt"
        runtime.optimizer.load_state_dict(
            torch.load(
                optimizer_path, map_location=runtime.distributed["device"], weights_only=True
            )
        )
        scheduler_path = checkpoint / "scheduler.pt"
        runtime.scheduler.load_state_dict(
            torch.load(scheduler_path, map_location="cpu", weights_only=True)
        )
        if int(meta.get("budget_update_equivalents", -1)) != (
            PARENT_UPDATE_EQUIVALENTS_SPENT + int(meta.get("new_global_updates", -1))
        ):
            raise FullVisualError("resume budget-equivalent counter differs")
    runtime.start_new_update = int(meta.get("new_global_updates", -1))
    runtime.start_anchor_presentations = int(meta.get("new_anchor_presentations", -1))
    runtime.budget_gpu_wall_seconds = float(meta.get("remote_gpu_wall_used_seconds", -1))
    runtime.finalization_reserve_seconds = float(
        meta.get("finalization_reserve_seconds", runtime.finalization_reserve_seconds)
    )
    runtime.preflight_receipt = meta.get("preflight_receipt")
    runtime.sampler.load_state(meta["sampler"])
    rank = int(runtime.distributed["rank"])
    rng_by_rank = meta.get("rng_by_rank") or []
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != 2:
        raise FullVisualError("resume checkpoint lacks per-rank RNG state")
    _restore_rng_state(rng_by_rank[rank], runtime.distributed["device"])
    runtime.checkpoint_meta = dict(meta)


# --------------------------------------------------------------------------- #
# Validation, status, and progress diagnostics (rank 0)
# --------------------------------------------------------------------------- #
def _teacher_energy(cache: ImmutableTargetCache, data: Any) -> dict[str, Any]:

    total_tokens = 0
    total_energy = 0.0
    anchors = 0
    for source in data.manifest.sources_for_split("validation"):
        source_id = str(source.source_demo_id)
        for endpoint in range(0, int(source.num_frames), 5):
            row = cache.read(source_id, endpoint)
            teacher = row.teacher_tokens.float()
            mask = (
                row.teacher_token_mask[:, None]
                .expand(teacher.shape[0], teacher.shape[1])
                .reshape(-1)
            )
            if not bool(mask.any()):
                raise FullVisualError("validation teacher row has no valid tokens")
            valid = int(mask.sum())
            total_energy += float(teacher.reshape(-1, teacher.shape[-1])[mask].square().sum())
            total_tokens += valid
            anchors += 1
    if total_tokens <= 0:
        raise FullVisualError("teacher energy denominator is empty")
    return {
        "anchors": anchors,
        "valid_tokens": total_tokens,
        "teacher_energy": total_energy / total_tokens,
    }


def _evaluate_validation(runtime: FullVisualRuntime) -> dict[str, Any]:
    from owac.evaluation.raw_temporal_features import evaluate_feature_validation

    student = runtime.ddp.module
    record = evaluate_feature_validation(
        student,
        runtime.dataset,
        runtime.cache,
        runtime.data,
        cosine_weight=float((runtime.config.get("training") or {}).get("cosine_weight", 0.1)),
        expected_sources=int(runtime.data_contract["expected_splits"]["validation"]),
        expected_windows=int(runtime.data_contract["expected_validation_windows"]),
    )
    energy = _teacher_energy(runtime.cache, runtime.data)
    relative = (
        float(record.feature_mse) / float(energy["teacher_energy"])
        if energy["teacher_energy"] > 0
        else None
    )
    payload = {
        "schema": "owac.raw-full-visual.validation.v1",
        "step": int(runtime.start_new_update),
        "budget_update_equivalents": int(
            PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
        ),
        "split": "validation",
        "sources": record.sources,
        "trajectories": record.trajectories,
        "windows": record.windows,
        "anchors": record.windows,
        "episode_resets": record.episode_resets,
        "valid_tokens": record.valid_tokens,
        "scalar_denominator": record.scalar_denominator,
        "feature_mse": record.feature_mse,
        "feature_cosine": record.feature_cosine,
        "feature_total": record.feature_total,
        "cosine_weight": record.cosine_weight,
        "teacher_energy": energy["teacher_energy"],
        "relative_feature_mse": relative,
        "per_source": {name: summary.to_payload() for name, summary in record.per_source.items()},
        "test_used": False,
        "selection_eligible": True,
        "timestamp": _now_iso(),
    }
    return payload


def _write_status(
    runtime: FullVisualRuntime,
    *,
    status: str,
    phase: str,
    last_checkpoint: Mapping[str, Any] | None = None,
    last_validation: Mapping[str, Any] | None = None,
    best: Mapping[str, Any] | None = None,
    error: str | None = None,
    gpu_wall_sum_seconds: float | None = None,
) -> None:
    if int(runtime.distributed["rank"]) != 0:
        return
    used_gpu = (
        float(runtime.budget_gpu_wall_seconds)
        if gpu_wall_sum_seconds is None
        else float(gpu_wall_sum_seconds)
    )
    payload = {
        "schema": "owac.raw-full-visual.run-status.v1",
        "status": status,
        "phase": phase,
        "pid": os.getpid(),
        "pid_start_ticks": _proc_start_ticks(os.getpid()),
        "pgid": os.getpgid(0),
        "started_at": runtime.report.get("started_at"),
        "updated_at": _now_iso(),
        "native_exit": None
        if status not in {"stopped", "completed", "error"}
        else (0 if status != "error" else 2),
        "cumulative_updates": int(runtime.start_new_update),
        "new_global_updates": int(runtime.start_new_update),
        "target_cumulative_updates": int(
            (runtime.config.get("training") or {}).get("max_new_global_updates", 3116)
        ),
        "budget_update_equivalents": int(
            PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
        ),
        "spent_update_equivalents_before_stage": PARENT_UPDATE_EQUIVALENTS_SPENT,
        "original_update_cap_equivalents": TOTAL_UPDATE_EQUIVALENTS,
        "anchor_presentations": int(runtime.start_anchor_presentations),
        "new_anchor_presentations": int(runtime.start_anchor_presentations),
        "target_anchor_presentations": int(
            (runtime.config.get("training") or {}).get("max_new_anchor_presentations", 49856)
        ),
        "budget_anchor_equivalents": int(
            PARENT_ANCHORS_EQUIVALENTS_SPENT + runtime.start_anchor_presentations
        ),
        "remote_gpu_wall_used_seconds": used_gpu,
        "remote_gpu_wall_cap_seconds": runtime.budget.remote_gpu_wall_cap_seconds,
        "total_gpu_wall_cap_seconds": runtime.budget.total_gpu_wall_seconds,
        "output_bytes": owned_run_bytes(runtime.run_root),
        "output_cap_bytes": runtime.budget.remote_output_cap_bytes,
        "last_checkpoint": None if last_checkpoint is None else dict(last_checkpoint),
        "last_validation": None if last_validation is None else dict(last_validation),
        "best": None if best is None else dict(best),
        "selection_metric": "validation_token_total",
        "error": error,
        "closed_loop": "not_run",
        "closed_loop_episodes": 0,
        "control_restored": False,
        "config_digest": _config_digest(runtime.config),
        "visual_source": runtime.report.get("source_checkpoint"),
        "visual_source_sha256": runtime.report.get("source_checkpoint_sha256"),
        "teacher_backend_optimizer_members": False,
        "full_visual_parameters": EXPECTED_FULL_VISUAL_PARAMETERS,
        "cache_rows": runtime.cache_report.get("rows"),
        "cache_load_seconds": runtime.cache_report.get("load_seconds"),
        "finalization_reserve_seconds": runtime.finalization_reserve_seconds,
        "data_protocol": runtime.data_contract.get("protocol"),
        "source_count": sum(len(values) for values in runtime.split.values()),
        "split": {name: len(values) for name, values in runtime.split.items()},
        "teacher_preparation_receipt_sha256": (
            None
            if runtime.teacher_prepare_receipt is None
            else runtime.teacher_prepare_receipt.get("sha256")
        ),
        "rank_processes": list(runtime.rank_processes),
        "gpu_by_rank": list(runtime.gpu_by_rank),
    }
    _atomic_json(runtime.run_root / "run-status.json", payload)
    _atomic_json(
        runtime.run_root / "heartbeat.json",
        {
            "schema": "owac.raw-full-visual.heartbeat.v1",
            "pid": os.getpid(),
            "pid_start_ticks": _proc_start_ticks(os.getpid()),
            "pgid": os.getpgid(0),
            "seq": 0,
            "updated_at": _now_iso(),
            "phase": phase,
            "new_global_updates": int(runtime.start_new_update),
        },
    )


def _write_run_meta(runtime: FullVisualRuntime) -> None:
    if int(runtime.distributed["rank"]) != 0:
        return
    payload = {
        "schema": "owac.raw-full-visual.run-meta.v1",
        "created_at": _now_iso(),
        "run_root": str(runtime.run_root),
        "config_digest": _config_digest(runtime.config),
        "mapped_config": runtime.mapped_config,
        "path_mapping": runtime.mapping_report,
        "identity": runtime.report,
        "cache_root": str(runtime.cache.root),
        "cache_identity_key": runtime.cache.identity_key,
        "cache_report": runtime.cache_report,
        "data_contract": runtime.data_contract,
        "teacher_prepare_receipt": runtime.teacher_prepare_receipt,
        "performance": runtime.performance_options.as_dict(),
        "fastdata": runtime.fastdata_report,
        "split": {name: list(values) for name, values in runtime.split.items()},
        "budget": {
            "prior_gpu_wall_seconds": runtime.budget.prior_gpu_wall_seconds,
            "total_gpu_wall_seconds": runtime.budget.total_gpu_wall_seconds,
            "remote_gpu_wall_cap_seconds": runtime.budget.remote_gpu_wall_cap_seconds,
            "prior_output_bytes": runtime.budget.prior_output_bytes,
            "total_output_bytes": runtime.budget.total_output_bytes,
            "remote_output_cap_bytes": runtime.budget.remote_output_cap_bytes,
            "spent_update_equivalents": runtime.budget.spent_update_equivalents,
            "total_update_equivalents": runtime.budget.total_update_equivalents,
            "new_updates_cap": runtime.budget.new_updates_cap,
            "spent_anchor_equivalents": runtime.budget.spent_anchor_equivalents,
            "total_anchor_equivalents": runtime.budget.total_anchor_equivalents,
            "new_anchors_cap": runtime.budget.new_anchors_cap,
        },
        "parent_provenance": dict(runtime.config.get("prior_stage") or {}),
        "parent_student_checkpoint_used": False,
        "teacher_backend_loaded_for_training": False,
        "distributed": {
            "world_size": int(runtime.distributed["world_size"]),
            "backend": runtime.distributed["backend"],
            "micro_batch_per_rank": int(
                (runtime.config.get("training") or {}).get("micro_batch_per_rank", 1)
            ),
            "gradient_accumulation_steps": int(
                (runtime.config.get("training") or {}).get("gradient_accumulation_steps", 8)
            ),
            "effective_batch": EXPECTED_EFFECTIVE_BATCH,
        },
    }
    _atomic_json(runtime.run_root / "run-meta.json", payload)


def _read_validation_history(run_root: Path) -> list[dict[str, Any]]:
    path = Path(run_root) / "validation.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _selector_from_validation_history(runtime: FullVisualRuntime) -> Any:
    from owac.evaluation.raw_temporal_features import FeatureSelector

    selector = FeatureSelector()
    for row in _read_validation_history(runtime.run_root):
        try:
            selector.consider(
                int(row["step"]),
                row,
                checkpoint_path=row.get("retained_model_path") or row.get("checkpoint_path"),
                checkpoint_sha256=row.get("checkpoint_sha256"),
            )
        except Exception:
            continue
    return selector


def _validation_and_checkpoint(
    runtime: FullVisualRuntime, *, kind: str, gpu_wall_sum_seconds: float | None = None
) -> dict[str, Any]:

    rank = int(runtime.distributed["rank"])
    validation = None
    selection_payload = None
    selector = _selector_from_validation_history(runtime)
    if rank == 0:
        validation = _evaluate_validation(runtime)
        improved = selector.consider(
            int(runtime.start_new_update),
            validation,
            checkpoint_path=None,
            checkpoint_sha256=None,
        )
        validation["is_best"] = bool(improved)
    _barrier()
    # Capture per-rank states and save only on rank 0.  The selector/checkpoint
    # SHA is filled after the model file is written.
    checkpoint_name = (
        "step0"
        if kind == "step0"
        else (
            f"checkpoint_final_{runtime.start_new_update:06d}"
            if kind == "final"
            else f"checkpoint_new_{runtime.start_new_update:06d}"
        )
    )
    checkpoint_meta = _save_full_visual_checkpoint(
        runtime,
        kind=kind,
        include_optimizer=kind != "step0",
        selection=None,
        validation=validation,
        gpu_wall_sum_seconds=(
            runtime.budget_gpu_wall_seconds
            if gpu_wall_sum_seconds is None
            else float(gpu_wall_sum_seconds)
        ),
        checkpoint_name=checkpoint_name,
    )
    if rank == 0:
        model_sha = checkpoint_meta["model_sha256"]
        checkpoint_rel = f"checkpoints/{checkpoint_name}"
        retained_model_path = "best/model.pt"
        best_path = None
        if validation is not None:
            validation["checkpoint_path"] = checkpoint_rel
            validation["historical_checkpoint_path"] = checkpoint_rel
            validation["checkpoint_sha256"] = model_sha
            if validation.get("is_best"):
                validation["retained_model_path"] = retained_model_path
                validation["retained_model_only"] = True
            _append_jsonl(runtime.run_root / "validation.jsonl", validation)
        if validation is not None and validation.get("is_best"):
            best_path = retained_model_path
        selection_payload = selector.sidecar(
            checkpoint_path=best_path,
            checkpoint_sha256=model_sha if best_path else None,
            updated_at=_now_iso(),
        )
        selection_payload["retained_model_path"] = retained_model_path
        selection_payload["retained_model_only"] = True
        selection_payload["source_checkpoint_path"] = None
        if selector.best_step is not None:
            history = _read_validation_history(runtime.run_root)
            best_row = next(
                (
                    row
                    for row in reversed(history)
                    if int(row.get("step", -1)) == int(selector.best_step)
                ),
                None,
            )
            if best_row is not None:
                selection_payload["source_checkpoint_path"] = best_row.get(
                    "historical_checkpoint_path"
                ) or best_row.get("checkpoint_path")
        _atomic_json(runtime.run_root / "selection.json", selection_payload)
        meta_path = runtime.run_root / checkpoint_rel / "meta.json"
        if meta_path.is_file():
            meta_update = _load_json(meta_path)
            meta_update["selection"] = selection_payload
            meta_update["validation"] = validation
            _atomic_json(meta_path, meta_update)
            _update_latest_pointer(runtime.run_root, runtime.run_root / checkpoint_rel, meta_update)
        last_checkpoint = {
            "step": int(runtime.start_new_update),
            "path": checkpoint_rel,
            "sha256": model_sha,
            "kind": kind,
        }
        _write_status(
            runtime,
            status="running" if kind != "step0" else "preparing",
            phase="full_visual_training",
            last_checkpoint=last_checkpoint,
            last_validation=validation,
            best=(
                None
                if not best_path
                else {
                    "step": int(runtime.start_new_update),
                    "value": validation["feature_total"],
                    "checkpoint_sha256": model_sha,
                    "path": best_path,
                    "retained_model_only": True,
                    "metric": "validation_token_total",
                }
            ),
        )
    _barrier()
    set_full_visual_train_mode(runtime.student, True)
    return checkpoint_meta


# --------------------------------------------------------------------------- #
# Training loop and preflight
# --------------------------------------------------------------------------- #
def _autocast_context(runtime: FullVisualRuntime) -> Any:
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _batched_group_loss(
    runtime: FullVisualRuntime,
    samples: Sequence[Any],
    weights: Sequence[float],
    token_counts: Sequence[int],
    component_records: list[dict[str, Any]],
) -> Any:
    import torch

    from owac.training.raw_distillation import token_distillation_loss
    from owac.training.raw_full_visual_batched import stack_teacher_targets

    with _autocast_context(runtime):
        result = runtime.adapter.forward_batched(tuple(samples))
    projected = result.projected_tokens
    batch = len(samples)
    if projected.shape[0] != batch or result.token_mask.shape[0] != batch:
        raise FullVisualError("batched student returned a different batch size")
    if projected.ndim != 3 or tuple(result.token_mask.shape) != tuple(projected.shape[:2]):
        raise FullVisualError("batched student projection/mask shape differs")
    teacher_flat, teacher_mask_flat = stack_teacher_targets(samples)
    teacher_flat = teacher_flat.to(projected.device)
    teacher_mask_flat = teacher_mask_flat.to(projected.device)
    if projected.shape[1] != teacher_flat.shape[1]:
        raise FullVisualError("batched student/teacher token width differs")
    combined = result.token_mask.to(dtype=torch.bool) & teacher_mask_flat.to(dtype=torch.bool)
    total_loss = None
    for index in range(batch):
        loss = token_distillation_loss(
            projected[index : index + 1],
            teacher_flat[index : index + 1],
            combined[index : index + 1],
            cosine_weight=float((runtime.config.get("training") or {}).get("cosine_weight", 0.1)),
        )
        if int(loss.valid_tokens) != int(token_counts[index]):
            raise FullVisualError(
                "batched forward token mask differs from the precomputed denominator"
            )
        component_records.append(
            {
                "mse": float(loss.mse.detach().float()),
                "cosine": float(loss.cosine.detach().float()),
                "total": float(loss.total.detach().float()),
                "tokens": int(loss.valid_tokens),
            }
        )
        if not bool(torch.isfinite(loss.total)):
            raise FullVisualError("non-finite batched distillation loss")
        weighted = loss.total * float(weights[index])
        if not bool(torch.isfinite(weighted)):
            raise FullVisualError("non-finite weighted batched distillation loss")
        total_loss = weighted if total_loss is None else total_loss + weighted
        if not bool(torch.isfinite(total_loss)):
            raise FullVisualError("non-finite accumulated batched distillation loss")
    if total_loss is None:
        raise FullVisualError("batched loss produced no samples")
    return total_loss


def _run_forward_backward(
    runtime: FullVisualRuntime,
    samples: Sequence[Any],
    weights: Sequence[float],
    token_counts: Sequence[int],
    component_records: list[dict[str, Any]],
) -> None:
    import contextlib

    import torch

    runtime.optimizer.zero_grad(set_to_none=True)
    if runtime.performance_options.batched_forward and len(samples) > 1:
        group_size = int(runtime.performance_options.per_rank_microbatch)
        if group_size not in {2, 4, 8}:
            raise FullVisualError("batched forward requires per-rank microbatch 2/4/8")
        groups = [
            list(samples[index : index + group_size])
            for index in range(0, len(samples), group_size)
        ]
        if len(groups) > 1 and len(groups[-1]) != group_size:
            raise FullVisualError("local sample count is not divisible by batched microbatch")
        for group_index, group in enumerate(groups):
            offset = group_index * group_size
            group_weights = weights[offset : offset + len(group)]
            group_counts = token_counts[offset : offset + len(group)]
            no_sync = group_index < len(groups) - 1
            context = runtime.ddp.no_sync() if no_sync else contextlib.nullcontext()
            with context:
                group_loss = _batched_group_loss(
                    runtime, group, group_weights, group_counts, component_records
                )
                group_loss.backward()
        return
    for index, (sample, weight, token_count) in enumerate(
        zip(samples, weights, token_counts, strict=True)
    ):
        context = runtime.ddp.no_sync() if index < len(samples) - 1 else contextlib.nullcontext()
        with context:
            with _autocast_context(runtime):
                result = runtime.trainer.forward_anchor(sample)
            if int(result["valid_tokens"]) != int(token_count):
                raise FullVisualError("forward token mask differs from the precomputed denominator")
            component_records.append(
                {
                    "mse": float(result["feature"].mse.detach().float()),
                    "cosine": float(result["feature"].cosine.detach().float()),
                    "total": float(result["feature"].total.detach().float()),
                    "tokens": int(token_count),
                }
            )
            loss = result["feature"].total * float(weight)
            if not bool(torch.isfinite(loss)):
                raise FullVisualError("non-finite distillation loss")
            loss.backward()


def run_training(
    config_path: str | Path,
    *,
    run_root: str | Path | None,
    max_new_updates: int | None,
    resume: str | Path | None,
    preflight_receipt: str | Path | None,
    performance_config: str | Path | None = None,
) -> dict[str, Any]:

    import torch
    import torch.distributed as dist

    invocation_started = time.monotonic()
    distributed = _init_distributed(dict(_load_json(config_path)))
    rank = int(distributed["rank"])
    world_size = int(distributed["world_size"])
    runtime = _build_runtime(
        config_path,
        run_root=run_root,
        distributed=distributed,
        resume_checkpoint=resume,
        preflight_receipt=preflight_receipt,
        performance_config=performance_config,
    )
    if not bool((runtime.config.get("authorization") or {}).get("optimizer_updates", False)):
        raise FullVisualError("optimizer updates are not authorized in this config")
    training = dict(runtime.config.get("training") or {})
    cap = int(training.get("max_new_global_updates", 3116))
    target = cap if max_new_updates is None else min(cap, int(max_new_updates))
    if target <= 0 or target > cap:
        raise FullVisualError("target new global update count is outside the cap")
    if runtime.start_new_update > target:
        raise FullVisualError("resume checkpoint is beyond the requested target")
    if runtime.start_anchor_presentations != runtime.start_new_update * EXPECTED_EFFECTIVE_BATCH:
        raise FullVisualError("resume anchor counter is inconsistent with the global batch")
    if runtime.start_new_update == 0 and runtime.preflight_receipt is None:
        raise FullVisualError("first train launch requires the real preflight receipt")
    startup_wall = _all_reduce_sum_float(time.monotonic() - invocation_started)
    runtime.budget_gpu_wall_seconds += startup_wall
    runtime.budget.assert_gpu_usage(runtime.budget_gpu_wall_seconds)
    runtime.rank_processes = _gather_rank_processes(distributed)
    runtime.gpu_by_rank = _gather_gpu_stats(distributed)
    stop_signal = {"value": False}

    def _handle_signal(_signum: int, _frame: Any) -> None:
        stop_signal["value"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    _write_run_meta(runtime)
    if rank == 0:
        _write_status(
            runtime,
            status="running",
            phase="full_visual_training",
            gpu_wall_sum_seconds=runtime.budget_gpu_wall_seconds,
        )
    try:
        if runtime.start_new_update >= target:
            # Strict reload with no further update: load already verified model,
            # optimizer, scheduler and per-rank RNG/sampler in _build_runtime.
            if rank == 0:
                _atomic_json(
                    runtime.run_root / "reload-check.json",
                    {
                        "schema": "owac.raw-full-visual.reload-check.v1",
                        "resume": None if resume is None else str(resume),
                        "new_global_updates": int(runtime.start_new_update),
                        "budget_update_equivalents": int(
                            PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
                        ),
                        "new_anchor_presentations": int(runtime.start_anchor_presentations),
                        "checkpoint_meta": runtime.checkpoint_meta,
                        "optimizer_loaded": runtime.checkpoint_meta is not None
                        and runtime.checkpoint_meta.get("optimizer_path") is not None,
                        "remote_gpu_wall_used_seconds": runtime.budget_gpu_wall_seconds,
                        "reload_zero_updates": True,
                        "timestamp": _now_iso(),
                    },
                )
                _write_status(
                    runtime,
                    status="completed",
                    phase="reload_verified",
                    gpu_wall_sum_seconds=runtime.budget_gpu_wall_seconds,
                )
            return {
                "status": "reload_verified",
                "new_global_updates": int(runtime.start_new_update),
                "budget_update_equivalents": int(
                    PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
                ),
                "new_anchor_presentations": int(runtime.start_anchor_presentations),
                "zero_optimizer_updates": True,
                "checkpoint": runtime.run_root / "reload-check.json",
            }
        # Step 0 is a real warm-start/audit checkpoint, model-only.
        phase_started = time.monotonic()
        phase_start_wall = runtime.budget_gpu_wall_seconds
        if runtime.start_new_update == 0:
            _validation_and_checkpoint(runtime, kind="step0", gpu_wall_sum_seconds=phase_start_wall)
        while runtime.start_new_update < target:
            stop_tensor = torch.tensor([1 if stop_signal["value"] else 0], device="cuda")
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            if int(stop_tensor.item()) > 0:
                break
            elapsed_local = time.monotonic() - phase_started
            max_elapsed = _all_reduce_max_float(elapsed_local)
            projected_gpu = phase_start_wall + world_size * max_elapsed
            if (
                projected_gpu + runtime.finalization_reserve_seconds
                > runtime.budget.remote_gpu_wall_cap_seconds
            ):
                break
            update_start = time.perf_counter()
            batch = runtime.sampler.next_batch()
            local_batch = rank_slice(batch, rank=rank, world_size=world_size)
            if len(local_batch) != EXPECTED_EFFECTIVE_BATCH // world_size:
                raise FullVisualError("rank local anchor count differs from the DDP plan")
            samples = [
                runtime.dataset.anchor_sample(source_id, endpoint)
                for source_id, endpoint in local_batch
            ]
            local_token_counts = [
                int(runtime.trainer._count_feature_tokens(sample)) for sample in samples
            ]
            if any(count <= 0 for count in local_token_counts):
                raise FullVisualError("local anchor has zero valid feature tokens")
            global_tokens = _all_reduce_int(sum(local_token_counts))
            if global_tokens <= 0:
                raise FullVisualError("global token denominator is zero")
            weights = global_accumulation_weights(
                local_token_counts, world_size=world_size, global_token_total=global_tokens
            )
            sample_construction_seconds = time.perf_counter() - update_start
            component_records: list[dict[str, Any]] = []
            identity_snapshot_seconds = 0.0
            forward_start = time.perf_counter()
            _run_forward_backward(runtime, samples, weights, local_token_counts, component_records)
            forward_backward_seconds = time.perf_counter() - forward_start
            grad_start = time.perf_counter()
            trainable = [p for p in runtime.student.parameters() if p.requires_grad]
            local_numerators = loss_component_numerators(component_records)
            numerator_tensor = torch.tensor(
                [
                    local_numerators["mse_numerator"],
                    local_numerators["cosine_numerator"],
                    local_numerators["total_numerator"],
                    local_numerators["tokens"],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            dist.all_reduce(numerator_tensor, op=dist.ReduceOp.SUM)
            global_loss_components = finalize_global_loss_components(
                {
                    "mse_numerator": float(numerator_tensor[0].item()),
                    "cosine_numerator": float(numerator_tensor[1].item()),
                    "total_numerator": float(numerator_tensor[2].item()),
                    "tokens": float(numerator_tensor[3].item()),
                },
                global_token_total=global_tokens,
            )
            torch.nn.utils.clip_grad_norm_(trainable, float(training.get("gradient_clip", 1.0)))
            lr_used = float(runtime.optimizer.param_groups[0]["lr"])
            runtime.optimizer.step()
            runtime.scheduler.step()
            next_lr = float(runtime.optimizer.param_groups[0]["lr"])
            runtime.optimizer.zero_grad(set_to_none=True)
            gradchecks_optimizer_seconds = time.perf_counter() - grad_start
            total_host_seconds = time.perf_counter() - update_start
            runtime.start_new_update += 1
            runtime.start_anchor_presentations += EXPECTED_EFFECTIVE_BATCH
            elapsed_local = time.monotonic() - phase_started
            wall_used = phase_start_wall + _all_reduce_sum_float(elapsed_local)
            runtime.budget.assert_gpu_usage(wall_used)
            runtime.budget_gpu_wall_seconds = wall_used
            if runtime.start_new_update % 10 == 0:
                runtime.gpu_by_rank = _gather_gpu_stats(distributed)
            if rank == 0:
                update_record = {
                    "schema": "owac.raw-full-visual.update.v1",
                    "event": "update",
                    "timestamp": _now_iso(),
                    "new_global_updates": int(runtime.start_new_update),
                    "budget_update_equivalents": int(
                        PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
                    ),
                    "new_anchor_presentations": int(runtime.start_anchor_presentations),
                    "sampled_source_endpoints": [
                        {"source_id": source_id, "endpoint": int(endpoint)}
                        for source_id, endpoint in batch
                    ],
                    "local_token_counts": local_token_counts,
                    "global_token_denominator": int(global_tokens),
                    "loss_components": global_loss_components,
                    "loss_component_accumulation": (
                        "sum(detached_fp32_component*token_count) over 8 local "
                        "microbatches; all_reduce SUM; divide global_tokens"
                    ),
                    "loss_scale_weights": weights,
                    "lr_used": lr_used,
                    "next_lr": next_lr,
                    "anchors": int(runtime.start_anchor_presentations),
                    "world_size": world_size,
                    "effective_batch": EXPECTED_EFFECTIVE_BATCH,
                    "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                    "gpu_by_rank": runtime.gpu_by_rank,
                    "rank_processes": runtime.rank_processes,
                    "remote_gpu_wall_used_seconds": wall_used,
                    "host_wall_phases": {
                        "sample_construction_seconds": sample_construction_seconds,
                        "identity_generation_setup_seconds": identity_snapshot_seconds,
                        "forward_backward_seconds": forward_backward_seconds,
                        "gradchecks_optimizer_seconds": gradchecks_optimizer_seconds,
                        "total_seconds": total_host_seconds,
                        "scope": "host_elapsed_including_waits_and_syncs_not_gpu_kernel_time",
                    },
                    "packet_cache": (
                        runtime.dataset.cache_report()
                        if hasattr(runtime.dataset, "cache_report")
                        else None
                    ),
                    "batch_scope": {
                        "global_effective_batch": EXPECTED_EFFECTIVE_BATCH,
                        "per_rank_microbatch": int(runtime.performance_options.per_rank_microbatch),
                        "batched_forward": bool(runtime.performance_options.batched_forward),
                        "gradient_accumulation_steps": (
                            EXPECTED_EFFECTIVE_BATCH
                            // world_size
                            // int(runtime.performance_options.per_rank_microbatch)
                        ),
                    },
                }
                _append_jsonl(runtime.run_root / "updates.jsonl", update_record)
                if runtime.start_new_update % 10 == 0:
                    _write_status(
                        runtime,
                        status="running",
                        phase="full_visual_training",
                        gpu_wall_sum_seconds=wall_used,
                    )
            _barrier()
            if runtime.start_new_update >= target:
                break
            if int(training.get("validation_interval_new_updates", 250)) > 0 and (
                runtime.start_new_update % int(training.get("validation_interval_new_updates", 250))
                == 0
            ):
                _validation_and_checkpoint(runtime, kind="interval", gpu_wall_sum_seconds=wall_used)
        # Final checkpoint always includes current optimizer state and true wall.
        if runtime.start_new_update > 0:
            elapsed_local = time.monotonic() - phase_started
            final_wall = phase_start_wall + _all_reduce_sum_float(elapsed_local)
            runtime.budget.assert_gpu_usage(final_wall)
            runtime.budget_gpu_wall_seconds = final_wall
            final = _validation_and_checkpoint(
                runtime, kind="final", gpu_wall_sum_seconds=final_wall
            )
            if rank == 0:
                _write_status(
                    runtime,
                    status="completed" if runtime.start_new_update >= target else "stopped",
                    phase="full_visual_training",
                    last_checkpoint={
                        "step": runtime.start_new_update,
                        "path": f"checkpoints/checkpoint_final_{runtime.start_new_update:06d}",
                        "sha256": final.get("model_sha256"),
                        "kind": "final",
                    },
                    gpu_wall_sum_seconds=final_wall,
                )
        return {
            "status": "completed" if runtime.start_new_update >= target else "stopped",
            "new_global_updates": int(runtime.start_new_update),
            "budget_update_equivalents": int(
                PARENT_UPDATE_EQUIVALENTS_SPENT + runtime.start_new_update
            ),
            "new_anchor_presentations": int(runtime.start_anchor_presentations),
            "remote_gpu_wall_used_seconds": runtime.budget_gpu_wall_seconds,
            "checkpoint": runtime.run_root / "latest.json",
        }
    except Exception as exc:
        if rank == 0:
            _write_status(
                runtime,
                status="error",
                phase="full_visual_training",
                error=f"{exc.__class__.__name__}: {exc}",
            )
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if dist.is_initialized():
            dist.destroy_process_group()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _measure_checkpoint_write(
    run_root: Path, *, bytes_to_write: int = 64 * 1024 * 1024
) -> dict[str, Any]:
    """Conservative local write benchmark for the final checkpoint reserve."""
    path = run_root / ".preflight-write.bin"
    chunk = b"\0" * (1024 * 1024)
    started = time.monotonic()
    with path.open("wb") as stream:
        remaining = int(bytes_to_write)
        while remaining > 0:
            piece = chunk[: min(len(chunk), remaining)]
            stream.write(piece)
            remaining -= len(piece)
        stream.flush()
        os.fsync(stream.fileno())
    elapsed = max(time.monotonic() - started, 1e-6)
    path.unlink(missing_ok=True)
    return {
        "bytes": int(bytes_to_write),
        "seconds": elapsed,
        "bytes_per_second": float(bytes_to_write) / elapsed,
        "fsynced": True,
        "temp_removed": not path.exists(),
    }


def preflight(
    config_path: str | Path,
    *,
    run_root: str | Path | None,
    performance_config: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    distributed = _init_distributed(dict(_load_json(config_path)))
    rank = int(distributed["rank"])
    started = time.monotonic()
    runtime = _build_runtime(
        config_path,
        run_root=run_root,
        distributed=distributed,
        performance_config=performance_config,
    )
    validation_started = time.monotonic()
    validation_payload = None
    if rank == 0:
        validation_payload = _evaluate_validation(runtime)
        set_full_visual_train_mode(runtime.student, True)
    _barrier()
    validation_wall = _all_reduce_max_float(time.monotonic() - validation_started)
    write_reserve = float(
        (runtime.config.get("training") or {}).get("checkpoint_write_reserve_seconds", 120.0)
    )
    write_benchmark = _measure_checkpoint_write(runtime.run_root) if rank == 0 else None
    checkpoint_bytes = int(sum(p.numel() * p.element_size() for p in runtime.student.parameters()))
    estimated_full_checkpoint_bytes = int(checkpoint_bytes * 3.05)
    estimated_write_seconds = estimated_full_checkpoint_bytes / max(
        1.0, float((write_benchmark or {}).get("bytes_per_second", 1.0))
    )
    runtime.finalization_reserve_seconds = max(
        float(runtime.finalization_reserve_seconds),
        validation_wall * 1.75 + max(write_reserve, estimated_write_seconds * 1.5) + 30.0,
    )
    elapsed_local = time.monotonic() - started
    wall_sum = _all_reduce_sum_float(elapsed_local)
    payload = None
    if rank == 0:
        payload = {
            "schema": PREFLIGHT_SCHEMA,
            "created_at": _now_iso(),
            "config_digest": _config_digest(runtime.config),
            "cache_identity_key": runtime.cache.identity_key,
            "optimizer_updates": 0,
            "trainable_parameters": int(
                sum(p.numel() for p in runtime.student.parameters() if p.requires_grad)
            ),
            "expected_trainable_parameters": EXPECTED_FULL_VISUAL_PARAMETERS,
            "scope": runtime.report.get("scope"),
            "preflight_phase_gpu_wall_seconds": wall_sum,
            "teacher_preparation_gpu_wall_seconds": float(runtime.budget_gpu_wall_seconds),
            "cumulative_gpu_wall_seconds": float(runtime.budget_gpu_wall_seconds) + wall_sum,
            "gpu_wall_sum_seconds": float(runtime.budget_gpu_wall_seconds) + wall_sum,
            "validation_wall_seconds": validation_wall,
            "validation": validation_payload,
            "finalization_reserve_seconds": runtime.finalization_reserve_seconds,
            "checkpoint_write_reserve_seconds": write_reserve,
            "checkpoint_write_benchmark": write_benchmark,
            "checkpoint_parameters_bytes": checkpoint_bytes,
            "estimated_full_checkpoint_bytes": estimated_full_checkpoint_bytes,
            "estimated_checkpoint_write_seconds": estimated_write_seconds,
            "cache_report": runtime.cache_report,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "world_size": int(distributed["world_size"]),
            "source": runtime.report,
            "data_contract": runtime.data_contract,
            "teacher_preparation_receipt": runtime.teacher_prepare_receipt,
            "performance": runtime.performance_options.as_dict(),
            "fastdata": runtime.fastdata_report,
            "split": {name: len(values) for name, values in runtime.split.items()},
            "parent_student_checkpoint_used": False,
            "teacher_backend_loaded": False,
            "preflight_receipt_path": str(runtime.run_root / "preflight.json"),
        }
        _write_run_meta(runtime)
        _atomic_json(runtime.run_root / "preflight.json", payload)
        _write_status(
            runtime,
            status="prepared",
            phase="preflight_complete",
            gpu_wall_sum_seconds=float(runtime.budget_gpu_wall_seconds) + wall_sum,
        )
    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return payload or {"status": "preflight_complete", "rank": rank}


def status(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root)
    path = root / "run-status.json"
    if not path.is_file():
        return {"status": "unknown", "run_root": str(root)}
    return _load_json(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--run-root", default=None)
    common.add_argument("--performance-config", default=None)
    pre = sub.add_parser("preflight", parents=[common])
    pre.add_argument("--device", default="cuda")
    train = sub.add_parser("train", parents=[common])
    train.add_argument("--max-new-updates", type=int, default=None)
    train.add_argument("--stop-after-new-update", type=int, default=None)
    train.add_argument("--resume", default=None)
    train.add_argument("--preflight-receipt", default=None)
    stat = sub.add_parser("status")
    stat.add_argument("--run-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(
                args.config,
                run_root=args.run_root,
                performance_config=args.performance_config,
            )
        elif args.command == "train":
            target = (
                args.stop_after_new_update
                if args.stop_after_new_update is not None
                else args.max_new_updates
            )
            result = run_training(
                args.config,
                run_root=args.run_root,
                max_new_updates=target,
                resume=args.resume,
                preflight_receipt=args.preflight_receipt,
                performance_config=args.performance_config,
            )
        else:
            result = status(args.run_root)
    except Exception as exc:
        traceback.print_exc()
        print(
            json.dumps(
                {"status": "unavailable", "reason": f"{exc.__class__.__name__}: {exc}"}, indent=2
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

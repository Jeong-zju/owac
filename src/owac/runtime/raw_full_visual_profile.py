# ruff: noqa: E501
"""CPU-only profiling of the full-visual data/packet path.

This probe deliberately does not construct the student, teacher, optimizer, or a
CUDA context.  It reads the active shared manifest and immutable target cache,
replays the actual 16 anchor identities recorded by the latest training update,
and times cold/warm RAW episode loading and packet construction for several
episode-cache sizes.

It never writes the shared dataset, source manifest, target cache, or live run
root.  Output is a JSON file in an explicitly requested, separate profile
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
import tracemalloc
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from owac.raw.shared_rawvla_distillation import load_shared_manifest
from owac.raw.teacher_data import TeacherData
from owac.raw.temporal_distillation import TemporalPacketDataset
from owac.runtime.raw_full_visual_distillation import ImmutableTargetCache

PROFILE_SCHEMA = "owac.raw-full-visual.data-profile.v1"
SHARED_PROTOCOL = "owac.shared-rawvla.task6-normal.8-2-2.v1"
TRAIN_ANCHOR_MIN = 25
ANCHOR_STRIDE = 5
EXPECTED_ANCHORS = 16


class ProfileError(RuntimeError):
    """The CPU profile input or safety contract is invalid."""


@dataclass(frozen=True, slots=True)
class AnchorIdentity:
    source_id: str
    endpoint: int


class _CPUStudent(torch.nn.Module):
    """Only supplies a CPU device through TemporalPacketDataset's parameter probe."""

    def __init__(self) -> None:
        super().__init__()
        self._device_anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))


@dataclass
class _PassStats:
    elapsed_seconds: float = 0.0
    anchor_seconds: list[float] | None = None
    episode_get: int = 0
    episode_hit: int = 0
    episode_miss: int = 0
    episode_put: int = 0
    npz_loads: int = 0
    npz_bytes: int = 0
    npz_seconds: float = 0.0
    target_reads: int = 0
    target_seconds: float = 0.0
    packet_builds: int = 0
    packet_seconds: float = 0.0
    peak_tracemalloc_bytes: int = 0
    cold: bool = False
    rss_before_bytes: int = 0
    rss_after_bytes: int = 0

    def __post_init__(self) -> None:
        if self.anchor_seconds is None:
            self.anchor_seconds = []

    def as_dict(self) -> dict[str, Any]:
        values = sorted(float(item) for item in (self.anchor_seconds or []))

        def percentile(fraction: float) -> float | None:
            if not values:
                return None
            index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
            return values[index]

        return {
            "elapsed_seconds": self.elapsed_seconds,
            "anchors": len(values),
            "anchor_ms_p50": None if not values else percentile(0.5) * 1000.0,
            "anchor_ms_p95": None if not values else percentile(0.95) * 1000.0,
            "episode_cache_get": self.episode_get,
            "episode_cache_hit": self.episode_hit,
            "episode_cache_miss": self.episode_miss,
            "episode_cache_put": self.episode_put,
            "npz_loads": self.npz_loads,
            "npz_decoded_bytes": self.npz_bytes,
            "npz_load_seconds": self.npz_seconds,
            "target_cache_reads": self.target_reads,
            "target_read_seconds": self.target_seconds,
            "packet_builds": self.packet_builds,
            "packet_build_seconds": self.packet_seconds,
            "tracemalloc_peak_bytes": self.peak_tracemalloc_bytes,
            "cold": self.cold,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
        }


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_last_update(
    path: Path, index: int, expected_anchors: int
) -> tuple[int, list[AnchorIdentity]]:
    if not path.is_file():
        raise ProfileError(f"updates.jsonl is absent: {path}")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("event") == "update":
            records.append(value)
    if not records:
        raise ProfileError("updates.jsonl has no update records")
    try:
        record = records[index]
    except IndexError as exc:
        raise ProfileError(f"update index {index} outside {len(records)} update records") from exc
    raw_anchors = record.get("sampled_source_endpoints")
    if not isinstance(raw_anchors, list) or len(raw_anchors) != int(expected_anchors):
        raise ProfileError(
            f"last update has {0 if not isinstance(raw_anchors, list) else len(raw_anchors)} "
            f"anchors, expected {expected_anchors}"
        )
    anchors = []
    for item in raw_anchors:
        if not isinstance(item, Mapping):
            raise ProfileError("recorded anchor is not an object")
        source_id = str(item.get("source_id") or "")
        try:
            endpoint = int(item.get("endpoint"))
        except (TypeError, ValueError) as exc:
            raise ProfileError("recorded anchor endpoint is invalid") from exc
        if not source_id or endpoint < TRAIN_ANCHOR_MIN or endpoint % ANCHOR_STRIDE:
            raise ProfileError(f"recorded anchor is not a training endpoint: {item!r}")
        anchors.append(AnchorIdentity(source_id, endpoint))
    return int(record.get("new_global_updates", 0)), anchors


def _validate_anchor_sources(anchors: Sequence[AnchorIdentity], manifest: Any) -> dict[str, int]:
    split_by_source = {source.source_demo_id: source.split for source in manifest.sources}
    counts = Counter()
    for anchor in anchors:
        split = split_by_source.get(anchor.source_id)
        if split is None:
            raise ProfileError(f"anchor source is not in shared manifest: {anchor.source_id}")
        if split != "train":
            raise ProfileError(
                f"anchor source {anchor.source_id} belongs to {split}, not train; "
                "validation/test observations must not be consumed"
            )
        counts[split] += 1
    return dict(counts)


def _packet_plan(anchors: Sequence[AnchorIdentity]) -> dict[str, Any]:
    keys: list[tuple[str, int]] = []
    unique_anchors = []
    for anchor in anchors:
        unroll = tuple(range(anchor.endpoint - 25, anchor.endpoint + 1, ANCHOR_STRIDE))
        prefix = tuple(range(0, unroll[0], ANCHOR_STRIDE))
        for endpoint in (*prefix, *unroll):
            keys.append((anchor.source_id, endpoint))
        unique_anchors.append(anchor)
    counts = Counter(keys)
    unique = list(counts)
    return {
        "total_packet_builds": len(keys),
        "unique_packet_keys": len(unique),
        "repeated_packet_builds": len(keys) - len(unique),
        "repeat_factor": (len(keys) / len(unique)) if unique else None,
        "by_source": dict(Counter(source for source, _ in keys)),
        "sources": len({anchor.source_id for anchor in anchors}),
        "unique_anchor_count": len(set(unique_anchors)),
    }


def _safe_output(path: Path, config: Mapping[str, Any]) -> Path:
    resolved = path.resolve()
    shared_root = Path(
        str((config.get("data") or {}).get("shared_root") or "/root/autodl-pub/datasets/rawvla")
    ).resolve()
    cache_root = Path(str((config.get("assets") or {}).get("cache_root") or "")).resolve()
    run_root = Path(str((config.get("remote") or {}).get("output_root") or "")).resolve()
    for label, root in (
        ("shared dataset", shared_root),
        ("target cache", cache_root),
        ("live run root", run_root),
    ):
        if str(root) and (resolved == root or root in resolved.parents):
            raise ProfileError(f"profile output must not be inside {label}: {resolved}")
    return resolved


def _cache_entries(raw: str) -> list[int]:
    values = sorted({int(item) for item in str(raw).split(",") if item.strip()})
    if not values or any(item < 1 for item in values):
        raise ProfileError("episode cache entries must be positive")
    return values


def _thread_values(raw: str) -> list[tuple[str, int | None]]:
    values: list[tuple[str, int | None]] = []
    for item in str(raw).split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"default", "auto"}:
            values.append(("default", None))
            continue
        try:
            count = int(token)
        except ValueError as exc:
            raise ProfileError(f"invalid torch thread count: {item!r}") from exc
        if count < 1:
            raise ProfileError("torch thread count must be positive")
        values.append((str(count), count))
    if not values:
        raise ProfileError("at least one torch thread mode is required")
    return values


def _run_pass(
    *,
    data: TeacherData,
    dataset: TemporalPacketDataset,
    cache: ImmutableTargetCache,
    anchors: Sequence[AnchorIdentity],
    cold: bool,
    tracemalloc_enabled: bool,
) -> _PassStats:
    stats = _PassStats()
    import owac.raw.teacher_data as teacher_module

    original_get = data.cache.get
    original_put = data.cache.put

    def timed_get(key: str) -> Any:
        stats.episode_get += 1
        value = original_get(key)
        if value is None:
            stats.episode_miss += 1
        else:
            stats.episode_hit += 1
        return value

    def timed_put(key: str, value: Any) -> None:
        stats.episode_put += 1
        original_put(key, value)

    data.cache.get = timed_get  # type: ignore[method-assign]
    data.cache.put = timed_put  # type: ignore[method-assign]

    original_npz = teacher_module._npz_file

    def timed_npz(path: Path, expected_sha: str) -> dict[str, Any]:
        start = time.perf_counter()
        result = original_npz(path, expected_sha)
        stats.npz_seconds += time.perf_counter() - start
        stats.npz_loads += 1
        stats.npz_bytes += sum(int(value.nbytes) for value in result.values())
        return result

    teacher_module._npz_file = timed_npz  # type: ignore[assignment]

    original_read = cache.read

    def timed_read(source_id: str, endpoint: int) -> Any:
        start = time.perf_counter()
        result = original_read(source_id, endpoint)
        stats.target_reads += 1
        stats.target_seconds += time.perf_counter() - start
        return result

    cache.read = timed_read  # type: ignore[method-assign]
    original_packet = dataset.packet_for_endpoint

    def timed_packet(source_id: str, endpoint: int, *, row: Any = None) -> Any:
        start = time.perf_counter()
        result = original_packet(source_id, endpoint, row=row)
        stats.packet_builds += 1
        stats.packet_seconds += time.perf_counter() - start
        return result

    dataset.packet_for_endpoint = timed_packet  # type: ignore[method-assign]

    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    if tracemalloc_enabled:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        for anchor in anchors:
            anchor_start = time.perf_counter()
            sample = dataset.anchor_sample(anchor.source_id, anchor.endpoint)
            if sample.target.source_id != anchor.source_id:
                raise ProfileError("reloaded target source identity differs")
            stats.anchor_seconds.append(time.perf_counter() - anchor_start)  # type: ignore[union-attr]
        stats.elapsed_seconds = time.perf_counter() - started
    finally:
        if tracemalloc_enabled:
            _, peak = tracemalloc.get_traced_memory()
            stats.peak_tracemalloc_bytes = int(peak)
            tracemalloc.stop()
        teacher_module._npz_file = original_npz  # type: ignore[assignment]
        data.cache.get = original_get  # type: ignore[method-assign]
        data.cache.put = original_put  # type: ignore[method-assign]
        cache.read = original_read  # type: ignore[method-assign]
        dataset.packet_for_endpoint = original_packet  # type: ignore[method-assign]
    stats.cold = bool(cold)
    stats.rss_before_bytes = rss_before
    stats.rss_after_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return stats


def _resolve_cache_identity(config: Mapping[str, Any]) -> str:
    cache_key = str((config.get("cache") or {}).get("identity_key") or "")
    if len(cache_key) == 64:
        return cache_key
    receipt_path = (config.get("remote") or {}).get("teacher_preparation_receipt")
    if receipt_path and Path(str(receipt_path)).is_file():
        receipt = json.loads(Path(str(receipt_path)).read_text(encoding="utf-8"))
        key = str(receipt.get("cache_identity_key") or "")
        if len(key) == 64:
            return key
    raise ProfileError("cannot resolve cache identity key")


def profile(
    *,
    config_path: str | Path,
    updates_path: str | Path,
    manifest_path: str | Path | None,
    cache_root_path: str | Path | None,
    output_path: str | Path,
    update_index: int = -1,
    expected_anchors: int = EXPECTED_ANCHORS,
    episode_cache_entries: str = "2,10",
    torch_threads: str = "8,default",
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    update_count, anchors = _load_last_update(
        Path(updates_path), int(update_index), int(expected_anchors)
    )
    manifest_path = Path(manifest_path or (config.get("assets") or {}).get("source_manifest"))
    cache_root = Path(cache_root_path or (config.get("assets") or {}).get("cache_root"))
    manifest = load_shared_manifest(manifest_path)
    _validate_anchor_sources(anchors, manifest)
    output = _safe_output(Path(output_path), config)
    cache_key = _resolve_cache_identity(config)
    cache = ImmutableTargetCache(cache_root, expected_identity_key=cache_key)

    preload_ids = [
        source.source_demo_id
        for split in ("train", "validation")
        for source in manifest.sources_for_split(split)
    ]
    preload_start = time.perf_counter()
    cache_report = cache.preload(preload_ids)
    preload_seconds = time.perf_counter() - preload_start

    selected_sources = {
        source.source_demo_id: source
        for source in manifest.sources
        if source.source_demo_id in {anchor.source_id for anchor in anchors}
    }
    original_threads = torch.get_num_threads()
    thread_modes = _thread_values(torch_threads)
    runs: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    try:
        for thread_label, thread_count in thread_modes:
            torch.set_num_threads(original_threads if thread_count is None else thread_count)
            effective_threads = torch.get_num_threads()
            mode_runs: dict[str, Any] = {}
            for entries in _cache_entries(episode_cache_entries):
                data = TeacherData(
                    manifest,
                    repo_root=Path(__file__).resolve().parents[3],
                    cache_entries=entries,
                )
                dataset = TemporalPacketDataset(
                    data=data, cache=cache, student=_CPUStudent(), arm="C"
                )
                cold = _run_pass(
                    data=data,
                    dataset=dataset,
                    cache=cache,
                    anchors=anchors,
                    cold=True,
                    tracemalloc_enabled=True,
                )
                warm = _run_pass(
                    data=data,
                    dataset=dataset,
                    cache=cache,
                    anchors=anchors,
                    cold=False,
                    tracemalloc_enabled=False,
                )
                mode_runs[str(entries)] = {
                    "cold": cold.as_dict(),
                    "warm": warm.as_dict(),
                    "cache_entries": entries,
                    "torch_num_threads": effective_threads,
                    "torch_thread_mode": thread_label,
                }
                runs[f"threads_{thread_label}_cache_{entries}"] = mode_runs[str(entries)]
            baseline = mode_runs.get("2")
            mode_comparisons: dict[str, Any] = {}
            if baseline:
                for entries, result in mode_runs.items():
                    if entries == "2":
                        continue
                    cold_speed = (
                        baseline["cold"]["elapsed_seconds"] / result["cold"]["elapsed_seconds"]
                        if result["cold"]["elapsed_seconds"] > 0
                        else None
                    )
                    warm_speed = (
                        baseline["warm"]["elapsed_seconds"] / result["warm"]["elapsed_seconds"]
                        if result["warm"]["elapsed_seconds"] > 0
                        else None
                    )
                    mode_comparisons[entries] = {
                        "cold_speedup_vs_2": cold_speed,
                        "warm_speedup_vs_2": warm_speed,
                        "npz_load_reduction_vs_2": (
                            baseline["cold"]["npz_loads"] - result["cold"]["npz_loads"]
                        ),
                    }
            comparisons[thread_label] = mode_comparisons
    finally:
        torch.set_num_threads(original_threads)
    plan = _packet_plan(anchors)
    result = {
        "schema": PROFILE_SCHEMA,
        "created_at": _now(),
        "config_path": str(Path(config_path).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "cache_root": str(cache_root.resolve()),
        "cache_identity_key": cache_key,
        "update_index": int(update_index),
        "new_global_updates": update_count,
        "protocol": (config.get("data") or {}).get("protocol"),
        "recorded_anchor_count": len(anchors),
        "recorded_anchors": [
            {"source_id": anchor.source_id, "endpoint": anchor.endpoint} for anchor in anchors
        ],
        "anchor_sources": len(selected_sources),
        "test_sources_used": False,
        "target_cache_preload": {
            **cache_report,
            "elapsed_seconds": preload_seconds,
            "read_only": True,
        },
        "episode_cache_entries": _cache_entries(episode_cache_entries),
        "torch_thread_modes": [label for label, _ in thread_modes],
        "torch_threads_before": original_threads,
        "torch_num_threads_after_restore": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "runs": runs,
        "comparisons_vs_cache_2": comparisons,
        "packet_plan": plan,
        "safety": {
            "cpu_only": True,
            "cuda_initialized": False,
            "optimizer_steps": 0,
            "student_or_teacher_built": False,
            "shared_dataset_written": False,
            "target_cache_written": False,
            "test_observations_used": False,
            "live_run_root_written": False,
            "episode_data_process_ram_only": True,
        },
        "limitations": [
            "This is a CPU data/packet profile, not a GPU utilization measurement.",
            "No student/teacher forward, CUDA context, optimizer, or checkpoint is created.",
            "The 16 anchor identities are the actual last-update sample; prefix/state semantics are unchanged.",
            "Packet construction is static RAW preprocessing and may be reusable within/strictly invalidation-bound; learned temporal state is never cached.",
            "Target cache rows are preloaded from the immutable 204-row cache; no test row is read.",
        ],
    }
    _write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--updates", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--update-index", type=int, default=-1)
    parser.add_argument("--expected-anchors", type=int, default=EXPECTED_ANCHORS)
    parser.add_argument("--episode-cache-entries", default="2,10")
    parser.add_argument("--torch-threads", default="8,default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import traceback

    try:
        result = profile(
            config_path=args.config,
            updates_path=args.updates,
            manifest_path=args.manifest,
            cache_root_path=args.cache_root,
            output_path=args.output,
            update_index=args.update_index,
            expected_anchors=args.expected_anchors,
            episode_cache_entries=args.episode_cache_entries,
            torch_threads=args.torch_threads,
        )
    except Exception as exc:
        traceback.print_exc()
        print(json.dumps({"status": "unavailable", "reason": f"{exc.__class__.__name__}: {exc}"}))
        return 2
    print(
        json.dumps(
            {
                "status": "profiled",
                "output": str(Path(args.output).resolve()),
                "new_global_updates": result["new_global_updates"],
                "anchors": result["recorded_anchor_count"],
                "cache_entries": result["episode_cache_entries"],
                "torch_thread_modes": result["torch_thread_modes"],
                "packet_plan": result["packet_plan"],
                "comparisons_vs_cache_2": result["comparisons_vs_cache_2"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

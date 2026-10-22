"""Opt-in CPU/GPU data-path acceleration for the full-visual stage.

The performance sidecar is deliberately separate from the mathematical experiment
config and config_digest.  Defaults are the legacy path: episode cache 2, no raw
preload, no identity snapshot, no packet cache and first-update-only audit clones
disabled (legacy behavior).  Enabling these options changes only caching/audit
mechanics, never sampler order, batch size, causal history, state recurrence,
loss, or optimizer math.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from owac.raw.teacher_data import TeacherData
from owac.raw.temporal_distillation import TemporalPacketDataset

PERFORMANCE_SCHEMA = "owac.raw-full-visual.performance.v1"
DEFAULT_EPISODE_CACHE_ENTRIES = 2
DEFAULT_PACKET_CACHE_BYTES = 0


class PerformanceConfigError(ValueError):
    """The opt-in performance sidecar is invalid."""


@dataclass(frozen=True, slots=True)
class PerformanceOptions:
    raw_preload: bool = False
    episode_cache_entries: int = DEFAULT_EPISODE_CACHE_ENTRIES
    packet_cache_max_bytes: int = DEFAULT_PACKET_CACHE_BYTES
    batched_forward: bool = False
    per_rank_microbatch: int = 1
    source_path: str = ""
    source_sha256: str | None = None
    options_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_preload": bool(self.raw_preload),
            "episode_cache_entries": int(self.episode_cache_entries),
            "packet_cache_max_bytes": int(self.packet_cache_max_bytes),
            "batched_forward": bool(self.batched_forward),
            "per_rank_microbatch": int(self.per_rank_microbatch),
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "options_sha256": self.options_sha256,
        }


def load_performance_options(path: str | Path | None) -> PerformanceOptions:
    if path is None:
        base = {
            "raw_preload": False,
            "episode_cache_entries": DEFAULT_EPISODE_CACHE_ENTRIES,
            "packet_cache_max_bytes": DEFAULT_PACKET_CACHE_BYTES,
            "batched_forward": False,
            "per_rank_microbatch": 1,
        }
        return PerformanceOptions(**base)
    source = Path(path)
    if not source.is_file():
        raise PerformanceConfigError(f"performance sidecar is absent: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PerformanceConfigError("performance sidecar is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise PerformanceConfigError("performance sidecar must be an object")
    if payload.get("schema") != PERFORMANCE_SCHEMA:
        raise PerformanceConfigError("performance sidecar schema differs")
    defaults = {
        "raw_preload": False,
        "episode_cache_entries": DEFAULT_EPISODE_CACHE_ENTRIES,
        "packet_cache_max_bytes": DEFAULT_PACKET_CACHE_BYTES,
        "batched_forward": False,
        "per_rank_microbatch": 1,
    }
    values = {}
    for key, default in defaults.items():
        value = payload.get(key, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise PerformanceConfigError(f"{key} must be boolean")
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise PerformanceConfigError(f"{key} must be an integer")
            if value < 0:
                raise PerformanceConfigError(f"{key} must be non-negative")
        values[key] = value
    if int(values["episode_cache_entries"]) < 1:
        raise PerformanceConfigError("episode_cache_entries must be positive")
    if int(values["per_rank_microbatch"]) not in {1, 2, 4, 8}:
        raise PerformanceConfigError("per_rank_microbatch must be one of 1, 2, 4, 8")
    if not values["batched_forward"] and int(values["per_rank_microbatch"]) != 1:
        raise PerformanceConfigError("per_rank_microbatch requires batched_forward=true")
    return PerformanceOptions(
        **values,
        source_path=str(source.resolve()),
        source_sha256=None,
        options_sha256=None,
    )


def _packet_bytes(packet: Any) -> int:
    size = 0
    for frame in getattr(packet, "frames", ()):
        values = getattr(frame, "values", None)
        if torch.is_tensor(values):
            size += int(values.numel()) * int(values.element_size())
    return size


class BoundedPacketCache:
    """LRU immutable packet cache; no learned state is ever stored here."""

    def __init__(self, max_bytes: int) -> None:
        if int(max_bytes) < 0:
            raise PerformanceConfigError("packet cache max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._sizes: dict[tuple[str, int], int] = {}
        self.current_bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0

    def get(self, key: tuple[str, int]) -> Any | None:
        value = self._items.get(key)
        if value is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: tuple[str, int], value: Any) -> bool:
        if self.max_bytes <= 0:
            return False
        size = _packet_bytes(value)
        if size <= 0 or size > self.max_bytes:
            return False
        if key in self._items:
            self._items.move_to_end(key)
            return False
        while self._items and self.current_bytes + size > self.max_bytes:
            old_key, _ = self._items.popitem(last=False)
            self.current_bytes -= self._sizes.pop(old_key, 0)
            self.evictions += 1
        self._items[key] = value
        self._sizes[key] = size
        self.current_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self.insertions += 1
        return True

    def report(self) -> dict[str, Any]:
        return {
            "max_bytes": int(self.max_bytes),
            "current_bytes": int(self.current_bytes),
            "peak_bytes": int(self.peak_bytes),
            "entries": len(self._items),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "insertions": int(self.insertions),
            "stores_learned_state": False,
            "key_semantics": "source_id + immutable endpoint",
        }


class CachedPacketDataset(TemporalPacketDataset):
    """TemporalPacketDataset with a bounded immutable packet cache."""

    def __init__(self, *, packet_cache_max_bytes: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.packet_cache = BoundedPacketCache(int(packet_cache_max_bytes))

    def packet_for_endpoint(self, source_id: str, endpoint: int, *, row: Any = None) -> Any:
        key = (str(source_id), int(endpoint))
        cached = self.packet_cache.get(key)
        if cached is not None:
            return cached
        packet = super().packet_for_endpoint(source_id, endpoint, row=row)
        self.packet_cache.put(key, packet)
        return packet

    def cache_report(self) -> dict[str, Any]:
        return self.packet_cache.report()


def preload_raw_episodes(data: TeacherData, manifest: Any) -> dict[str, Any]:
    """Fill episode LRU with train+validation RAW once; never test sources."""
    started = time.perf_counter()
    loaded = 0
    source_ids = []
    for split in ("train", "validation"):
        for source in manifest.sources_for_split(split):
            source_id = str(source.source_demo_id)
            source_ids.append(source_id)
            data.episode_raw(source_id)
            loaded += 1
    return {
        "sources": loaded,
        "source_ids": source_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "test_sources_loaded": 0,
        "writes_shared": False,
        "cache_entries": int(getattr(data.cache, "max_entries", 0)),
    }

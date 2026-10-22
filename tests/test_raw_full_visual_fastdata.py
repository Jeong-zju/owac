"""CPU tests for opt-in full-visual data-path performance options."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from owac.runtime.raw_full_visual_fastdata import (
    BoundedPacketCache,
    PerformanceConfigError,
    load_performance_options,
    preload_raw_episodes,
)


def test_performance_defaults_are_legacy_and_sidecar_is_separate(tmp_path: Path) -> None:
    defaults = load_performance_options(None)
    assert defaults.raw_preload is False
    assert defaults.episode_cache_entries == 2
    assert defaults.packet_cache_max_bytes == 0
    sidecar = tmp_path / "performance.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema": "owac.raw-full-visual.performance.v1",
                "raw_preload": True,
                "episode_cache_entries": 10,
                "packet_cache_max_bytes": 1_500_000_000,
            }
        ),
        encoding="utf-8",
    )
    options = load_performance_options(sidecar)
    assert options.raw_preload is True
    assert options.episode_cache_entries == 10
    assert options.packet_cache_max_bytes == 1_500_000_000
    nd = options.as_dict()
    assert nd["raw_preload"] is True
    assert nd["options_sha256"] is None


def test_performance_sidecar_rejects_bad_schema_and_values(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "wrong", "episode_cache_entries": 10}), encoding="utf-8")
    with pytest.raises(PerformanceConfigError, match="schema"):
        load_performance_options(path)
    path.write_text(
        json.dumps(
            {
                "schema": "owac.raw-full-visual.performance.v1",
                "episode_cache_entries": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PerformanceConfigError, match="positive"):
        load_performance_options(path)


def _packet(values: torch.Tensor, *, reset: bool = False) -> SimpleNamespace:
    frame = SimpleNamespace(values=values)
    return SimpleNamespace(
        frames=(frame,),
        source_demo_id="source-a",
        endpoint=0,
        frame_ids=(0,),
        padding_mask=(False,),
        reset=reset,
        preprocess_id="test-preprocess",
        camera_order=("base",),
        measurement_metadata={"tag": "native"},
        timestamps=None,
    )


def test_bounded_packet_cache_is_lru_exact_and_state_free() -> None:
    cache = BoundedPacketCache(max_bytes=64)
    left = _packet(torch.zeros(1, 1, 3, 4, 4))
    right = _packet(torch.ones(1, 1, 3, 4, 4))
    # 1*1*3*4*4*4 = 192 bytes per packet > 64: neither fits.
    assert cache.put(("a", 0), left) is False
    assert cache.max_bytes == 64
    assert cache.current_bytes == 0
    assert cache.get(("a", 0)) is None

    small = BoundedPacketCache(max_bytes=1024)
    assert small.put(("a", 0), left) is True
    assert small.get(("a", 0)) is left
    assert small.get(("a", 0)).measurement_metadata == {"tag": "native"}
    assert small.put(("b", 0), right) is True
    report = small.report()
    assert report["stores_learned_state"] is False
    assert report["key_semantics"] == "source_id + immutable endpoint"
    assert report["current_bytes"] <= report["max_bytes"]


def test_preload_raw_excludes_test_and_never_writes() -> None:
    calls = []

    class FakeCache:
        max_entries = 10

    class FakeData:
        cache = FakeCache()

        def episode_raw(self, source_id):
            calls.append(source_id)
            return SimpleNamespace(source_demo_id=source_id)

    train = [SimpleNamespace(source_demo_id="train-1"), SimpleNamespace(source_demo_id="train-2")]
    validation = [SimpleNamespace(source_demo_id="val-1")]
    test = [SimpleNamespace(source_demo_id="test-1")]
    manifest = SimpleNamespace(
        sources_for_split=lambda split: {
            "train": train,
            "validation": validation,
            "test": test,
        }[split]
    )
    report = preload_raw_episodes(FakeData(), manifest)
    assert calls == ["train-1", "train-2", "val-1"]
    assert report["sources"] == 3
    assert report["test_sources_loaded"] == 0
    assert report["writes_shared"] is False
    assert report["cache_entries"] == 10


def test_cached_packet_returns_exact_native_packet_tensors_and_metadata(monkeypatch) -> None:
    from owac.raw.temporal_distillation import TemporalPacketDataset
    from owac.runtime.raw_full_visual_fastdata import CachedPacketDataset

    frame_values = torch.arange(1 * 1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 1, 3, 8, 8)
    native = SimpleNamespace(
        frames=(SimpleNamespace(values=frame_values),),
        source_demo_id="source-a",
        endpoint=0,
        frame_ids=(0,),
        padding_mask=(False,),
        reset=True,
        timestamps=None,
        camera_order=("base",),
        preprocess_id="native-preprocess",
        measurement_metadata={"mask": (True,), "timing": "native"},
    )
    monkeypatch.setattr(
        TemporalPacketDataset,
        "packet_for_endpoint",
        lambda self, source_id, endpoint, row=None: native,
    )
    dataset = object.__new__(CachedPacketDataset)
    dataset.packet_cache = BoundedPacketCache(max_bytes=1_000_000)
    first = dataset.packet_for_endpoint("source-a", 0)
    second = dataset.packet_for_endpoint("source-a", 0)
    assert first is native and second is native
    assert second.frames[0].values is frame_values
    assert torch.equal(second.frames[0].values, native.frames[0].values)
    assert second.frame_ids == (0,)
    assert second.padding_mask == (False,)
    assert second.reset is True
    assert second.camera_order == ("base",)
    assert second.preprocess_id == "native-preprocess"
    assert second.measurement_metadata == {"mask": (True,), "timing": "native"}
    assert dataset.packet_cache.report()["hits"] == 1


def test_checked_in_performance_sidecar_is_valid_and_separate() -> None:
    path = Path("configs/experiments/raw_full_visual_performance_v1.json")
    options = load_performance_options(path)
    assert options.raw_preload is True
    assert options.episode_cache_entries == 10
    assert options.packet_cache_max_bytes == 1_500_000_000
    assert options.batched_forward is True
    assert options.per_rank_microbatch == 8
    assert options.options_sha256 is None

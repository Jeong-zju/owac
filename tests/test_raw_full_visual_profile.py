"""Focused CPU-only tests for the full-visual data/packet profiler."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from owac.runtime.raw_full_visual_profile import (
    AnchorIdentity,
    ProfileError,
    _cache_entries,
    _load_last_update,
    _packet_plan,
    _run_pass,
    _safe_output,
    _thread_values,
    _validate_anchor_sources,
)


def _anchors(count: int = 16) -> list[dict[str, object]]:
    return [
        {"source_id": f"shared_task6_normal_demo_{4 + index}", "endpoint": 30 + index * 5}
        for index in range(count)
    ]


def test_load_actual_shaped_update_anchors(tmp_path: Path) -> None:
    updates = tmp_path / "updates.jsonl"
    updates.write_text(
        "\n".join(
            [
                json.dumps({"event": "start"}),
                json.dumps(
                    {
                        "event": "update",
                        "new_global_updates": 32,
                        "sampled_source_endpoints": _anchors(),
                    }
                ),
                json.dumps({"event": "status"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    step, anchors = _load_last_update(updates, -1, 16)
    assert step == 32
    assert len(anchors) == 16
    assert anchors[0] == AnchorIdentity("shared_task6_normal_demo_4", 30)
    with pytest.raises(ProfileError, match="expected 15"):
        _load_last_update(updates, -1, 15)


def test_anchor_source_must_be_train_and_known() -> None:
    manifest = SimpleNamespace(
        sources=[
            SimpleNamespace(source_demo_id="train", split="train"),
            SimpleNamespace(source_demo_id="val", split="validation"),
            SimpleNamespace(source_demo_id="test", split="test"),
        ]
    )
    assert _validate_anchor_sources([AnchorIdentity("train", 30)], manifest) == {"train": 1}
    with pytest.raises(ProfileError, match="belongs to validation"):
        _validate_anchor_sources([AnchorIdentity("val", 30)], manifest)
    with pytest.raises(ProfileError, match="not in shared manifest"):
        _validate_anchor_sources([AnchorIdentity("missing", 30)], manifest)


def test_packet_plan_quantifies_repeated_prefix_builds() -> None:
    plan = _packet_plan(
        [
            AnchorIdentity("shared_task6_normal_demo_1", 30),
            AnchorIdentity("shared_task6_normal_demo_1", 40),
        ]
    )
    assert plan["total_packet_builds"] == 16
    assert plan["unique_packet_keys"] == 9
    assert plan["repeated_packet_builds"] == 7
    assert plan["repeat_factor"] == pytest.approx(16 / 9)


def test_output_cannot_be_written_into_shared_cache_or_live_run(tmp_path: Path) -> None:
    config = {
        "data": {"shared_root": str(tmp_path / "shared")},
        "assets": {"cache_root": str(tmp_path / "cache")},
        "remote": {"output_root": str(tmp_path / "live")},
    }
    for root in ("shared", "cache", "live"):
        with pytest.raises(ProfileError, match="must not be inside"):
            _safe_output(tmp_path / root / "profile.json", config)
    allowed = _safe_output(tmp_path / "profile" / "profile.json", config)
    assert allowed == (tmp_path / "profile/profile.json").resolve()


def test_cache_entries_are_positive_and_deduplicated() -> None:
    assert _cache_entries("12,2,10,2") == [2, 10, 12]
    with pytest.raises(ProfileError):
        _cache_entries("0,2")


def test_cpu_profile_instrumentation_has_no_model_or_cuda_dependency() -> None:
    class FakeEpisodeCache:
        def __init__(self) -> None:
            self.values = {}
            self.get_calls = 0

        def get(self, key):
            self.get_calls += 1
            return self.values.get(key)

        def put(self, key, value):
            self.values[key] = value

    class FakeData:
        def __init__(self) -> None:
            self.cache = FakeEpisodeCache()

    class FakeTargetCache:
        def __init__(self) -> None:
            self.reads = 0

        def read(self, source_id, endpoint):
            self.reads += 1
            return {"source_id": source_id, "endpoint": endpoint}

    class FakeDataset:
        def __init__(self, data, cache) -> None:
            self.data = data
            self.cache = cache

        def packet_for_endpoint(self, source_id, endpoint, *, row=None):
            return (source_id, endpoint)

        def anchor_sample(self, source_id, endpoint):
            self.cache.read(source_id, endpoint)
            self.data.cache.get(f"{source_id}:{endpoint}")
            self.packet_for_endpoint(source_id, endpoint)
            return SimpleNamespace(target=SimpleNamespace(source_id=source_id))

    data = FakeData()
    target_cache = FakeTargetCache()
    dataset = FakeDataset(data, target_cache)
    stats = _run_pass(
        data=data,
        dataset=dataset,
        cache=target_cache,
        anchors=[AnchorIdentity("shared_task6_normal_demo_1", 30)],
        cold=True,
        tracemalloc_enabled=False,
    )
    assert stats.cold is True
    assert stats.target_reads == 1
    assert stats.packet_builds == 1
    assert stats.episode_get == 1
    assert stats.episode_miss == 1
    assert stats.anchor_seconds and len(stats.anchor_seconds) == 1


def test_torch_thread_modes_are_explicit_and_restorable() -> None:
    assert _thread_values("8,default") == [("8", 8), ("default", None)]
    assert _thread_values("10") == [("10", 10)]
    with pytest.raises(ProfileError):
        _thread_values("0")
    with pytest.raises(ProfileError):
        _thread_values("")

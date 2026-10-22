import json

import numpy as np
import torch

from owac.raw.all_bench_distillation import (
    DOMAIN_CAMERAS,
    TargetStore,
    load_inventory,
    normalize_raw,
)


def test_inventory_grouping_and_split(tmp_path):
    inv = {
        "groups": [
            {
                "group_id": "g1",
                "domain": "libero",
                "suite": "s",
                "task_name": "t",
                "split": "train",
                "num_frames": 30,
                "lighting": "Normal",
                "relpath": "x.npz",
                "path": "x.npz",
            },
            {
                "group_id": "g2",
                "domain": "robotwin2",
                "suite": "r",
                "task_name": "r",
                "split": "validation",
                "num_frames": 30,
                "lighting": "Low",
                "relpath": "y.npz",
                "path": "y.npz",
            },
        ]
    }
    p = tmp_path / "i.json"
    p.write_text(json.dumps(inv))
    sources, _ = load_inventory(p)
    assert [s.group_id for s in sources] == ["g1", "g2"]
    assert DOMAIN_CAMERAS["libero"] == ("base_0_rgb", "left_wrist_0_rgb")
    assert len(DOMAIN_CAMERAS["robotwin2"]) == 3


def test_libero_uint8_and_robotwin_float_normalization_are_not_same_rule():
    lib = np.full((2, 256, 256, 3), 255, dtype=np.uint8)
    out = normalize_raw(lib, "libero")
    assert torch.allclose(out, torch.ones_like(out))
    rt = np.full((2, 240, 320, 3), 0.4, dtype=np.float32)
    out = normalize_raw(rt, "robotwin2")
    assert out.shape == (2, 3, 256, 256)
    assert float(out.max()) <= 0.40001 and float(out.max()) >= 0.39999


def test_tokens_only_cache_roundtrip(tmp_path):
    store = TargetStore(tmp_path)
    npy, meta = store.paths("g1")
    data = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(2, 3, 256, 2048))
    data[0] = 1
    data[1] = 2
    data.flush()
    del data
    meta.write_text(
        json.dumps({"schema": "x", "complete": True, "windows": 2, "endpoints": [0, 5]})
    )
    assert store.exists("g1")
    row = store.read("g1", 5)
    assert row.teacher_tokens.shape == (3, 256, 2048)
    assert float(row.teacher_tokens.mean()) == 2.0
    assert row.teacher_token_mask.all()


def test_epoch_chunks_are_domain_homogeneous_and_preserve_all_anchors():
    from pathlib import Path

    from owac.raw.all_bench_distillation import BenchSource, _chunks_for_epoch

    sources = {
        "l1": BenchSource("l1", "libero", "s", "t", "train", Path("x"), 30, "Normal", ("a", "b")),
        "l2": BenchSource("l2", "libero", "s", "t", "train", Path("x"), 30, "Normal", ("a", "b")),
        "r1": BenchSource(
            "r1", "robotwin2", "r", "t", "train", Path("y"), 30, "Normal", ("a", "b", "c")
        ),
    }
    keys = [(k, e) for k in sources for e in (25, 30) if e < sources[k].num_frames]
    chunks = _chunks_for_epoch(keys, sources, seed=101, epoch=0)
    assert all(len(c) == 16 for c in chunks)
    assert max(len({sources[k].domain for k, _ in c}) for c in chunks) == 1


def test_cpu_sample_is_224_six_frame_reset_and_device_movable(tmp_path):
    import json

    import numpy as np
    import torch

    from owac.raw.all_bench_distillation import AllBenchDataset, BenchSource, TargetStore

    raw = np.zeros((2, 31, 256, 256, 3), dtype=np.uint8)
    raw[0, :, 0, 0] = 255
    raw[1, :, 1, 1] = 255
    p = tmp_path / "raw.npz"
    np.savez_compressed(p, agentview_raw_uint8=raw[0], wrist_raw_uint8=raw[1])
    src = BenchSource(
        "g", "libero", "s", "t", "train", p, 31, "Normal", ("base_0_rgb", "left_wrist_0_rgb")
    )
    store = TargetStore(tmp_path / "targets")
    npy, meta = store.paths("g")
    arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(2, 2, 256, 2048))
    arr[0] = 1
    arr[1] = 2
    arr.flush()
    del arr
    meta.write_text(json.dumps({"complete": True, "windows": 2, "endpoints": [0, 25]}))
    ds = AllBenchDataset([src], targets=store)
    sample = ds.sample("g", 25, device=None)
    assert sample.unroll_packets[-1].frames[0].values.shape == (1, 2, 3, 224, 224)
    assert len(sample.unroll_packets) == 6 and sample.unroll_packets[0].reset
    moved = ds.sample("g", 25, device=torch.device("cpu"))
    assert moved.target.teacher_tokens.device.type == "cpu"


def test_anchor_keys_rejects_missing_targets(tmp_path):
    import pytest

    from owac.raw.all_bench_distillation import AllBenchDataset, BenchSource, TargetStore

    src = BenchSource(
        "missing", "libero", "s", "t", "train", tmp_path / "no.npz", 31, "Normal", ("a", "b")
    )
    ds = AllBenchDataset([src], targets=TargetStore(tmp_path / "targets"))
    with pytest.raises(FileNotFoundError):
        ds.anchor_keys("train")


def test_best_selection_chronology_and_atomic_checkpoint(tmp_path):
    import torch

    from owac.raw.all_bench_distillation import atomic_save, select_best

    best, improved = select_best(None, 1.25)
    assert best == 1.25 and improved
    best, improved = select_best(best, 1.50)
    assert best == 1.25 and not improved
    best, improved = select_best(best, 0.80)
    assert best == 0.80 and improved
    path = tmp_path / "latest" / "train.pt"
    atomic_save(path, {"step": 250, "best_total": best, "rng_by_rank": ["r0", "r1"]})
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["step"] == 250 and loaded["best_total"] == 0.80
    assert not list(path.parent.glob("*.tmp"))


def test_stacked_burst_mover_preserves_geometry_and_masks_on_cpu():
    import torch

    from owac.baselines.raw_stem_siglip import RawStemInput
    from owac.training.raw_full_visual_batched import move_raw_input_to_device

    values = torch.zeros(6, 2, 3, 224, 224)
    mask = torch.ones(6, 2, 224, 224, dtype=torch.bool)
    item = RawStemInput(
        values=values,
        input_mode="rawvla_rgb3",
        camera_ids=("a", "b"),
        units="normalized_0_1",
        valid_mask=mask,
    )
    moved = move_raw_input_to_device(item, torch.device("cpu"))
    assert moved.values.shape == (6, 2, 3, 224, 224) and moved.valid_mask.shape == mask.shape


def test_full_sample_prefetch_preserves_order_and_runs_workers():
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from owac.raw.all_bench_distillation import submit_cpu_samples

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_loader(source_id, endpoint):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return (source_id, endpoint)

    keys = [("s0", 25), ("s1", 25), ("s2", 25), ("s3", 25)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = submit_cpu_samples(executor, fake_loader, keys)
        result = [future.result() for future in futures]
    assert result == keys
    assert peak >= 2


def test_working_cache_capacity_wiring_and_bounded_retention(tmp_path, monkeypatch):
    from pathlib import Path

    import owac.raw.all_bench_distillation as module

    config = {"training": {"working_cache_sources": 16}}
    capacity = int(config["training"]["working_cache_sources"])
    source = module.BenchSource(
        "g", "libero", "s", "t", "train", Path("x"), 31, "Normal", ("a", "b")
    )
    dataset = module.AllBenchDataset(
        [source], targets=module.TargetStore(tmp_path), capacity=capacity
    )
    assert dataset.cache.capacity == 16
    monkeypatch.setattr(module, "Episode", lambda item: object())
    sources = {
        f"g{i}": module.BenchSource(
            f"g{i}", "libero", "s", "t", "train", Path("x"), 31, "Normal", ("a", "b")
        )
        for i in range(20)
    }
    cache = module.EpisodeCache(sources, capacity=16)
    for source_id in sources:
        cache.get(source_id)
    assert len(cache._items) <= 16


def test_first_full_checkpoint_and_periodic_predicate():
    from owac.raw.all_bench_distillation import should_save_checkpoint

    assert should_save_checkpoint(1)
    assert not should_save_checkpoint(2)
    assert should_save_checkpoint(250)
    assert should_save_checkpoint(500)


def test_npz_stored_memmap_selected_frames_and_compressed_fallback(tmp_path):
    import zipfile

    import numpy as np

    from owac.raw.all_bench_distillation import BenchSource, Episode, read_npz_member

    raw = np.arange(12 * 256 * 256 * 3, dtype=np.uint8).reshape(12, 256, 256, 3) % 251
    wrist = np.flip(raw, axis=0).copy()
    stored = tmp_path / "stored.npz"
    np.savez(stored, agentview_raw_uint8=raw, wrist_raw_uint8=wrist)
    with zipfile.ZipFile(stored) as archive:
        assert archive.getinfo("agentview_raw_uint8.npy").compress_type == zipfile.ZIP_STORED
    member = read_npz_member(stored, "agentview_raw_uint8.npy")
    assert isinstance(member, np.memmap) and not member.flags.writeable
    assert member.dtype == raw.dtype and member.shape == raw.shape
    assert np.array_equal(np.asarray(member[3:9]), raw[3:9])

    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(compressed, agentview_raw_uint8=raw, wrist_raw_uint8=wrist)
    fallback = read_npz_member(compressed, "agentview_raw_uint8.npy")
    assert not isinstance(fallback, np.memmap)
    assert np.array_equal(fallback, raw)

    src = BenchSource(
        "stored",
        "libero",
        "s",
        "t",
        "train",
        stored,
        12,
        "Normal",
        ("base_0_rgb", "left_wrist_0_rgb"),
    )
    episode = Episode(src)
    burst = episode.burst(5)
    assert burst.shape == (2, 6, 3, 256, 256)
    expected = np.stack([raw[i] for i in range(6)]).transpose(0, 3, 1, 2) / 255.0
    assert np.allclose(burst[0].numpy(), expected)

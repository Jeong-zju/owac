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


def test_target_store_open_copy_close_repeated_and_concurrent(tmp_path):
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch

    from owac.raw.all_bench_distillation import TargetStore

    store = TargetStore(tmp_path)
    ids = [f"s{i}" for i in range(20)]
    for i, source_id in enumerate(ids):
        npy, meta = store.paths(source_id)
        data = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(1, 2, 3, 8))
        data[0] = float(i)
        data.flush()
        del data
        meta.write_text(json.dumps({"complete": True, "windows": 1, "endpoints": [0]}))

    first = store.read(ids[0], 0)
    old = first.teacher_tokens.clone()
    before = len(os.listdir("/proc/self/fd"))
    for source_id in ids + ids:
        store.read(source_id, 0)
    torch.testing.assert_close(first.teacher_tokens, old)
    assert len(os.listdir("/proc/self/fd")) <= before + 2

    def work(source_id):
        return float(store.read(source_id, 0).teacher_tokens.mean())

    with ThreadPoolExecutor(max_workers=4) as executor:
        values = list(executor.map(work, ids * 3))
    assert len(values) == 60
    torch.testing.assert_close(first.teacher_tokens, old)


def test_validation_interval_and_epoch_boundary_scheduling():
    from owac.raw.all_bench_distillation import should_validate

    assert not should_validate(1, epoch_end=False, interval=500)
    assert not should_validate(499, epoch_end=False, interval=500)
    assert should_validate(500, epoch_end=False, interval=500)
    assert should_validate(1000, epoch_end=False, interval=500)
    assert should_validate(17, epoch_end=True, interval=500)  # epoch end
    assert should_validate(500, epoch_end=True, interval=500)  # coincident bound, one call
    assert not should_validate(500, epoch_end=False, interval=0)


def test_prefix_recurrence_matches_reference_camera_masks_and_reset():
    import torch

    from owac.training.raw_full_visual_batched import prefix_gate_step

    torch.manual_seed(7)
    cameras, patches, width = 2, 3, 4
    alpha = torch.rand(1, 1, width)
    hidden = None
    for _step, reset in enumerate((True, False, False, True)):
        mean = torch.randn(cameras, patches, width)
        support = torch.tensor([[True, False, True], [True, True, False]])
        present = torch.tensor([True, False])
        current = prefix_gate_step(mean, support, present, hidden, alpha, reset)
        fresh = torch.where(support.unsqueeze(-1), mean, torch.zeros_like(mean))
        if hidden is None or reset:
            expected = fresh
        else:
            advanced = (1.0 - alpha) * mean + alpha * hidden
            recurrent = torch.where(support.unsqueeze(-1), advanced, hidden)
            expected = torch.where(present[:, None, None], recurrent, hidden)
        torch.testing.assert_close(current, expected)
        hidden = current


def test_serial_validation_end_to_end_metrics_and_distinct_cosine(monkeypatch):
    import contextlib
    from types import SimpleNamespace

    import torch

    import owac.raw.all_bench_distillation as module

    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())

    from dataclasses import dataclass

    @dataclass
    class FakeFrame:
        values: torch.Tensor

    @dataclass
    class FakePacket:
        frames: tuple

    class FakeStudent:
        training = True

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, packet, state=None):
            return SimpleNamespace(
                projected_tokens=torch.ones(1, 4, 4),
                tokens=SimpleNamespace(mask=torch.ones(1, 4, dtype=torch.bool)),
            ), state

    sources = [
        SimpleNamespace(
            split="validation", group_id="a", domain="libero", lighting="Normal", endpoints=(0, 25)
        ),
        SimpleNamespace(
            split="validation", group_id="b", domain="robotwin2", lighting="Low", endpoints=(0, 25)
        ),
    ]
    targets = {}
    for source in sources:
        targets[source.group_id] = {
            ep: SimpleNamespace(
                teacher_tokens=torch.zeros(2, 2, 4),
                teacher_token_mask=torch.ones(2, dtype=torch.bool),
            )
            for ep in source.endpoints
        }

    sources_map = {source.group_id: source for source in sources}

    class Dataset:
        sources = sources_map
        targets = SimpleNamespace(read=lambda sid, ep: targets[sid][ep])

        def packet_for_endpoint(self, sid, ep, device=None):
            return FakePacket(
                frames=tuple(FakeFrame(torch.zeros(1, 2, 3, 224, 224)) for _ in range(6))
            )

    result = module.validate_all_bench(FakeStudent(), Dataset(), device=torch.device("cpu"))
    assert result["windows"] == 4
    assert result["valid_tokens"] == 16
    assert result["feature_mse"] == 1.0
    assert result["feature_cosine"] == 1.0
    assert abs(result["feature_total"] - 1.1) < 1e-6
    assert result["feature_cosine"] != result["feature_total"]
    assert result["startup"]["windows"] == 2
    assert result["eligible_ge_25"]["windows"] == 2
    assert set(result["per_domain"]) == {"libero", "robotwin2"}
    assert set(result["per_lighting"]) == {"Normal", "Low"}


def test_whole_prefix_two_step_chunk_matches_original_sequential_reference(monkeypatch):
    from types import SimpleNamespace

    import torch

    import owac.training.raw_full_visual_batched as batched

    cameras, patches, width = 2, 3, 4

    def fake_burst(student, packets):
        means = []
        supports = []
        presents = []
        for packet in packets:
            ep = int(packet.endpoint)
            means.append(
                torch.arange(cameras * patches * width, dtype=torch.float32).reshape(
                    cameras, patches, width
                )
                / (ep + 1)
            )
            supports.append(
                torch.tensor([[True, (ep % 2) == 0, True], [True, True, (ep % 3) != 0]])
            )
            presents.append(torch.tensor([True, (ep % 4) != 0]))
        return (torch.stack(means), torch.stack(supports), torch.stack(presents), None, None)

    monkeypatch.setattr(batched, "_burst_components", fake_burst)

    class Student:
        gate_logit = torch.zeros(width)
        gate_alpha = torch.full((width,), 0.25)
        geometry = SimpleNamespace(num_patches=patches, hidden_size=width)

    def packet(ep, reset):
        return SimpleNamespace(
            endpoint=ep,
            reset=reset,
            source_demo_id="s",
            frames=(SimpleNamespace(camera_ids=("a", "b")),),
        )

    samples = []
    for length, reset_at in ((1, 0), (3, 0), (2, 0)):
        prefix = [packet(i * 5, i == reset_at) for i in range(length)]
        unroll = [packet(25 + 5 * i, i == 0) for i in range(6)]
        samples.append(SimpleNamespace(prefix_packets=prefix, unroll_packets=unroll))
    got, got_has = batched._prefix_hidden(Student(), samples)

    expected = torch.zeros((len(samples), cameras, patches, width))
    expected_has = torch.zeros(len(samples), dtype=torch.bool)
    for sample_index, sample in enumerate(samples):
        hidden = None
        for step_index in range(len(sample.prefix_packets)):
            pkt = sample.prefix_packets[step_index]
            mean, support, present, _, _ = fake_burst(Student(), [pkt])
            hidden = batched.prefix_gate_step(
                mean[0],
                support[0],
                present[0],
                hidden,
                Student().gate_alpha.reshape(1, 1, -1),
                bool(pkt.reset),
            )
            expected_has[sample_index] = True
        if hidden is not None:
            expected[sample_index] = hidden
    torch.testing.assert_close(got, expected)
    assert torch.equal(got_has, expected_has)


def test_sample_frame_cache_reuses_storage_numeric_and_chunks(tmp_path, monkeypatch):
    import json

    import numpy as np
    import torch

    from owac.raw import all_bench_distillation as module
    from owac.raw.all_bench_distillation import AllBenchDataset, BenchSource, TargetStore

    frames = np.arange(36 * 256 * 256, dtype=np.uint32).reshape(36, 256, 256, 1)
    frames = ((frames * 7) % 251).astype(np.uint8).repeat(3, axis=3)
    wrist = np.flip(frames, axis=0).copy()
    raw = tmp_path / "raw.npz"
    np.savez(raw, agentview_raw_uint8=frames, wrist_raw_uint8=wrist)
    source = BenchSource(
        "g", "libero", "s", "t", "train", raw, 36, "Normal", ("base_0_rgb", "left_wrist_0_rgb")
    )
    targets = TargetStore(tmp_path / "targets")
    npy, meta = targets.paths("g")
    data = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(1, 2, 3, 8))
    data[0] = 3
    data.flush()
    del data
    meta.write_text(json.dumps({"complete": True, "windows": 1, "endpoints": [35]}))
    dataset = AllBenchDataset([source], targets=targets)

    calls = {"normalize": 0, "interpolate": 0}
    real_normalize = module.normalize_raw
    real_interpolate = module.F.interpolate

    def count_normalize(values, domain):
        calls["normalize"] = max(calls["normalize"], int(values.shape[0]))
        return real_normalize(values, domain)

    def count_interpolate(values, *args, **kwargs):
        calls["interpolate"] = max(calls["interpolate"], int(values.shape[0]))
        return real_interpolate(values, *args, **kwargs)

    monkeypatch.setattr(module, "normalize_raw", count_normalize)
    monkeypatch.setattr(module.F, "interpolate", count_interpolate)

    sample = dataset.sample("g", 35, device=None)
    assert calls["normalize"] <= 16 and calls["interpolate"] <= 16
    for packet in sample.unroll_packets + sample.prefix_packets:
        old = dataset.packet_for_endpoint("g", packet.endpoint)
        for new_frame, old_frame in zip(packet.frames, old.frames, strict=True):
            torch.testing.assert_close(new_frame.values, old_frame.values)

    # Overlapping endpoint bursts share the exact physical frame storage.
    shared_new = sample.prefix_packets[-1].frames[-1].values.data_ptr()
    shared_unroll = sample.unroll_packets[0].frames[0].values.data_ptr()
    assert shared_new == shared_unroll

    unique = {}
    duplicated = 0
    for packet in sample.unroll_packets + sample.prefix_packets:
        for frame in packet.frames:
            pointer = frame.values.data_ptr()
            duplicated += frame.values.numel() * frame.values.element_size()
            unique[pointer] = frame.values.numel() * frame.values.element_size()
    assert sum(unique.values()) < duplicated


def test_robotwin_three_camera_nonconstant_values_match_old_path(tmp_path):
    import json

    import numpy as np
    import torch

    from owac.raw.all_bench_distillation import AllBenchDataset, BenchSource, TargetStore

    t = np.linspace(0.0, 0.4, 36 * 240 * 320 * 3, dtype=np.float32).reshape(36, 240, 320, 3)
    raw = tmp_path / "r.npz"
    np.savez(
        raw,
        head_camera_raw_float32=t,
        left_camera_raw_float32=(t * 0.5),
        right_camera_raw_float32=(t * 0.25),
    )
    source = BenchSource(
        "r",
        "robotwin2",
        "task",
        "task",
        "train",
        raw,
        36,
        "Normal",
        ("robotwin_head", "robotwin_left", "robotwin_right"),
    )
    targets = TargetStore(tmp_path / "t")
    npy, meta = targets.paths("r")
    data = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(1, 3, 3, 8))
    data[0] = 1
    data.flush()
    del data
    meta.write_text(json.dumps({"complete": True, "windows": 1, "endpoints": [35]}))
    dataset = AllBenchDataset([source], targets=targets)
    sample = dataset.sample("r", 35)
    for packet in sample.unroll_packets + sample.prefix_packets:
        old = dataset.packet_for_endpoint("r", packet.endpoint)
        for new_frame, old_frame in zip(packet.frames, old.frames, strict=True):
            torch.testing.assert_close(new_frame.values, old_frame.values)
    assert sample.unroll_packets[-1].frames[0].values.shape == (1, 3, 3, 224, 224)


def test_robotwin_three_camera_burst_shapes_and_frame_reuse(tmp_path, monkeypatch):
    import json

    import numpy as np

    from owac.raw.all_bench_distillation import AllBenchDataset, BenchSource, TargetStore

    frames = np.full((31, 240, 320, 3), 0.4, dtype=np.float32)
    raw = tmp_path / "r.npz"
    np.savez(
        raw,
        head_camera_raw_float32=frames,
        left_camera_raw_float32=frames,
        right_camera_raw_float32=frames,
    )
    source = BenchSource(
        "r",
        "robotwin2",
        "task",
        "task",
        "train",
        raw,
        31,
        "Normal",
        ("robotwin_head", "robotwin_left", "robotwin_right"),
    )
    targets = TargetStore(tmp_path / "t")
    npy, meta = targets.paths("r")
    data = np.lib.format.open_memmap(npy, mode="w+", dtype=np.float32, shape=(1, 3, 3, 8))
    data[0] = 1
    data.flush()
    del data
    meta.write_text(json.dumps({"complete": True, "windows": 1, "endpoints": [25]}))
    dataset = AllBenchDataset([source], targets=targets)
    sample = dataset.sample("r", 25)
    assert sample.unroll_packets[-1].frames[0].values.shape == (1, 3, 3, 224, 224)
    assert len(sample.unroll_packets) == 6
    assert sample.unroll_packets[0].padding_mask[0] is True

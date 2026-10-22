"""CPU contracts for the remote full-visual two-GPU distillation stage."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from owac.runtime.raw_full_visual_distillation import (
    EXPECTED_FULL_VISUAL_PARAMETERS,
    FullVisualBudget,
    FullVisualError,
    PathMapper,
    classify_full_visual_parameter,
    deterministic_source_split,
    global_accumulation_weights,
    load_safetensors_subset,
    owned_run_bytes,
    parse_safetensors_header,
    plan_checkpoint_retention,
    rank_slice,
    select_full_visual_trainable,
    validate_checkpoint_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/experiments/raw_stem_siglip_full_visual_v1.json"


class _FakeParameter:
    def __init__(self, numel: int):
        self._numel = int(numel)

    def numel(self) -> int:
        return self._numel


def _fake_named_parameters() -> dict:
    counts = {
        "gate": 1152,
        "stem": 14453632,
        "position": 294912,
        "blocks": 411466608,
        "norm": 2304,
        "projector": 2361344,
    }
    names = {
        "gate": "gate_logit",
        "stem": "base.vision_tower.vision_model.embeddings.patch_embedding.conv64.weight",
        "position": "base.vision_tower.vision_model.embeddings.position_embedding.weight",
        "blocks": "base.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        "norm": "base.vision_tower.vision_model.post_layernorm.weight",
        "projector": "base.projector.linear.weight",
    }
    return {names[key]: _FakeParameter(value) for key, value in counts.items()}


def test_budget_matches_exact_remaining_stage_authorization() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    budget = FullVisualBudget.from_config(config)
    assert budget.remaining_update_equivalents == 3116
    assert budget.remaining_anchor_equivalents == 49856
    assert budget.remote_gpu_wall_cap_seconds == pytest.approx(140841.2631939833)
    assert budget.remote_output_cap_bytes == 15396584412
    assert budget.prior_gpu_wall_seconds == pytest.approx(89558.73680601672)
    assert budget.prior_output_bytes == 6078252068


def test_full_visual_scope_and_parameter_count_contract() -> None:
    selected = select_full_visual_trainable(_fake_named_parameters())
    assert selected["total"] == EXPECTED_FULL_VISUAL_PARAMETERS
    assert selected["matches_expected"] is True
    assert selected["rejected"] == []
    assert selected["categories"] == {
        "gate": 1152,
        "stem": 14453632,
        "position": 294912,
        "blocks": 411466608,
        "norm": 2304,
        "projector": 2361344,
    }
    assert classify_full_visual_parameter("gate_logit") == "gate"
    assert (
        classify_full_visual_parameter(
            "base.vision_tower.vision_model.encoder.layers.26.mlp.fc1.weight"
        )
        == "blocks"
    )
    assert classify_full_visual_parameter("base.projector.linear.weight") == "projector"
    assert classify_full_visual_parameter("base.language_model.weight") is None


def test_ddp_weights_sum_to_world_size_and_ranks_are_disjoint() -> None:
    weights = global_accumulation_weights([100, 50], world_size=2, global_token_total=150)
    assert weights == pytest.approx([4.0 / 3.0, 2.0 / 3.0])
    assert sum(weights) == pytest.approx(2.0)
    batch = [f"anchor-{index}" for index in range(16)]
    left = rank_slice(batch, rank=0, world_size=2)
    right = rank_slice(batch, rank=1, world_size=2)
    assert len(left) == len(right) == 8
    assert not set(left) & set(right)
    assert left + right == batch
    with pytest.raises(FullVisualError):
        global_accumulation_weights([0], world_size=2, global_token_total=10)


def test_deterministic_split_requires_40_5_5_and_no_overlap() -> None:
    train = tuple(SimpleNamespace(source_demo_id=f"train/{i}") for i in range(40))
    validation = tuple(SimpleNamespace(source_demo_id=f"val/{i}") for i in range(5))
    test = tuple(SimpleNamespace(source_demo_id=f"test/{i}") for i in range(5))
    manifest = SimpleNamespace(
        sources_for_split=lambda split: {
            "train": train,
            "validation": validation,
            "test": test,
        }[split]
    )
    split = deterministic_source_split(manifest)
    assert len(split["train"]) == 40
    assert len(split["validation"]) == 5
    assert len(split["test"]) == 5
    bad = SimpleNamespace(sources_for_split=lambda split: train if split == "train" else validation)
    with pytest.raises(FullVisualError, match="validation and test"):
        deterministic_source_split(bad)


def test_path_mapping_preserves_logical_provenance(tmp_path: Path) -> None:
    physical = tmp_path / "root" / "repo" / "assets" / "file.bin"
    physical.parent.mkdir(parents=True)
    physical.write_bytes(b"asset")
    mapper = PathMapper(
        [{"logical_prefix": "/home/example/repo", "physical_prefix": str(tmp_path / "root/repo")}]
    )
    resolved = mapper.resolve("/home/example/repo/assets/file.bin")
    assert resolved.physical == str(physical)
    assert resolved.logical == "/home/example/repo/assets/file.bin"
    assert resolved.mapped is True
    report = mapper.report(["/home/example/repo/assets/file.bin"])
    assert report[0]["mapped"] is True


def test_tiny_safetensors_subset_loader(tmp_path: Path) -> None:
    tensor = [1.0, 2.0, 3.0, 4.0]
    data = struct.pack("<4f", *tensor)
    header = {
        "module.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, len(data)],
        }
    }
    encoded = json.dumps(header).encode("utf-8")
    path = tmp_path / "tiny.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    parsed, data_start = parse_safetensors_header(path)
    assert parsed == header and data_start == 8 + len(encoded)
    loaded = load_safetensors_subset(
        path, key_map={"module.weight": "vision_model.weight"}, dtype=__import__("torch").float32
    )
    assert list(loaded) == ["vision_model.weight"]
    assert loaded["vision_model.weight"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_checkpoint_retention_cap_and_unique_inode_accounting(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "model.pt").write_bytes(b"x" * 10)
    os.link(root / "model.pt", root / "best-model.pt")
    assert owned_run_bytes(root) == 10
    plan = plan_checkpoint_retention(
        15_000_000_000, cap_bytes=15_396_584_412, reserve_bytes=5_200_000_000
    )
    assert plan["fits"] is False
    small = plan_checkpoint_retention(
        2_000_000_000, cap_bytes=15_396_584_412, reserve_bytes=5_200_000_000
    )
    assert small["fits"] is True
    assert small["max_full_checkpoints"] == 2


def _checkpoint_meta(**overrides):
    meta = {
        "schema": "owac.raw-full-visual.checkpoint.v1",
        "kind": "step0",
        "scope": "full_visual_head",
        "world_size": 2,
        "model_sha256": "a" * 64,
        "checkpoint_sha256": "a" * 64,
        "optimizer_path": None,
        "optimizer_sha256": None,
        "new_global_updates": 0,
        "new_anchor_presentations": 0,
        "budget_update_equivalents": 1884,
        "config_digest": "b" * 64,
        "source_sha256": "c" * 64,
        "parent_student_checkpoint_used": False,
        "rng_by_rank": [{}, {}],
        "sampler": {"batch_size": 16},
    }
    meta.update(overrides)
    return meta


def test_checkpoint_contract_step0_and_full() -> None:
    step0 = validate_checkpoint_metadata(
        _checkpoint_meta(), config_digest="b" * 64, source_sha256="c" * 64
    )
    assert step0["kind"] == "step0" and step0["optimizer_present"] is False
    full = _checkpoint_meta(
        kind="interval",
        new_global_updates=250,
        new_anchor_presentations=4000,
        budget_update_equivalents=2134,
        optimizer_path="optimizer.pt",
        optimizer_sha256="d" * 64,
    )
    result = validate_checkpoint_metadata(full, config_digest="b" * 64, source_sha256="c" * 64)
    assert result["optimizer_present"] is True
    with pytest.raises(FullVisualError, match="model-only"):
        validate_checkpoint_metadata(
            _checkpoint_meta(optimizer_path="optimizer.pt"),
            config_digest="b" * 64,
            source_sha256="c" * 64,
        )
    with pytest.raises(FullVisualError, match="anchor counter"):
        validate_checkpoint_metadata(
            _checkpoint_meta(new_anchor_presentations=1),
            config_digest="b" * 64,
            source_sha256="c" * 64,
        )
    with pytest.raises(FullVisualError, match="old student"):
        validate_checkpoint_metadata(
            _checkpoint_meta(parent_student_checkpoint_used=True),
            config_digest="b" * 64,
            source_sha256="c" * 64,
        )


def test_config_frozen_stage_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["student"]["trainable_parameters"] == EXPECTED_FULL_VISUAL_PARAMETERS
    assert config["student"]["trainable_scope"] == "full_visual_head"
    assert config["training"]["learning_rate"] == 1e-5
    assert config["training"]["effective_batch"] == 16
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["max_new_global_updates"] == 3116
    assert config["training"]["max_new_anchor_presentations"] == 49856
    assert config["prior_stage"]["parent_student_checkpoint_used"] is False
    assert config["remote"]["world_size"] == 2
    assert config["evaluation"]["validation_windows"] == 102
    assert config["evaluation"]["test_selection"] is False
    assert config["cache"]["expected_rows"] == 969
    assert config["cache"]["expected_train_anchors"] == 667


class _TinyGateModel(__import__("torch").nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = __import__("torch").nn.Linear(3, 1, bias=True)
        self.gate = __import__("torch").nn.Parameter(__import__("torch").zeros(1))

    def forward(self, value):
        return self.stem(value) * __import__("torch").sigmoid(self.gate)


def _ddp_weight_worker(rank, world_size, init_file, queue):
    import contextlib

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=int(rank),
        world_size=int(world_size),
    )
    torch.manual_seed(101)
    model = DDP(_TinyGateModel(), find_unused_parameters=False)
    inputs = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3) / 100.0
    targets = torch.linspace(-1.0, 1.0, 16).reshape(16, 1)
    tokens = torch.tensor(
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], dtype=torch.float32
    )
    global_tokens = float(tokens.sum())
    local_indices = list(range(rank * 8, (rank + 1) * 8))
    model.zero_grad(set_to_none=True)
    for index, item in enumerate(local_indices):
        context = model.no_sync() if index < len(local_indices) - 1 else contextlib.nullcontext()
        with context:
            prediction = model(inputs[item : item + 1])
            loss = (prediction - targets[item : item + 1]).square() * (
                float(world_size) * float(tokens[item]) / global_tokens
            )
            loss.backward()
    queue.put(
        {
            "rank": int(rank),
            "stem_weight_grad": model.module.stem.weight.grad.detach().cpu().tolist(),
            "gate_grad": model.module.gate.grad.detach().cpu().tolist(),
        }
    )
    dist.destroy_process_group()


def test_two_rank_ddp_weighting_matches_serial_global_batch(tmp_path: Path) -> None:
    import multiprocessing as mp

    import torch

    init_file = tmp_path / "ddp-init"
    context = mp.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_ddp_weight_worker,
            args=(rank, 2, str(init_file), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0, process.exitcode
    rank_gradients = {
        int(item["rank"]): item for item in (queue.get(timeout=30), queue.get(timeout=30))
    }
    assert set(rank_gradients) == {0, 1}
    rank0_stem = torch.tensor(rank_gradients[0]["stem_weight_grad"], dtype=torch.float32)
    rank1_stem = torch.tensor(rank_gradients[1]["stem_weight_grad"], dtype=torch.float32)
    rank0_gate = torch.tensor(rank_gradients[0]["gate_grad"], dtype=torch.float32)
    rank1_gate = torch.tensor(rank_gradients[1]["gate_grad"], dtype=torch.float32)
    assert torch.allclose(rank0_stem, rank1_stem, atol=1e-7, rtol=1e-7)
    assert torch.allclose(rank0_gate, rank1_gate, atol=1e-7, rtol=1e-7)

    torch.manual_seed(101)
    serial = _TinyGateModel()
    inputs = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3) / 100.0
    targets = torch.linspace(-1.0, 1.0, 16).reshape(16, 1)
    tokens = torch.tensor(
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47], dtype=torch.float32
    )
    serial.zero_grad(set_to_none=True)
    loss = 0.0
    for index in range(16):
        prediction = serial(inputs[index : index + 1])
        loss = loss + (prediction - targets[index : index + 1]).square() * (
            float(tokens[index]) / float(tokens.sum())
        )
    loss.backward()
    ddp_stem = rank0_stem
    ddp_gate = rank0_gate
    assert torch.allclose(serial.stem.weight.grad, ddp_stem, atol=1e-6, rtol=1e-6)
    assert torch.allclose(serial.gate.grad, ddp_gate, atol=1e-6, rtol=1e-6)


def test_cache_preload_is_explicit_and_memoized(tmp_path: Path) -> None:
    from owac.runtime.raw_full_visual_distillation import ImmutableTargetCache

    cache = ImmutableTargetCache.__new__(ImmutableTargetCache)
    cache.root = tmp_path
    cache.identity_key = "a" * 64
    cache._rows = {}
    for source_id in ("train-a", "validation-a"):
        directory = cache.root / cache.identity_key / source_id
        directory.mkdir(parents=True)
        for endpoint in (0, 5):
            (directory / f"{endpoint:06d}.pt").write_bytes(b"row")
    cache.cached_endpoints = lambda source_id: (0, 5) if source_id != "test" else (0,)
    loads = []

    def fake_read(source_id, endpoint):
        loads.append((source_id, endpoint))
        return {"source_id": source_id, "endpoint": endpoint}

    cache.read = fake_read
    report = cache.preload(["train-a", "validation-a"])
    assert report["rows"] == 4
    assert report["sources"] == 2
    assert report["memoized"] is True
    assert report["test_rows_preloaded"] is False
    assert loads == [
        ("train-a", 0),
        ("train-a", 5),
        ("validation-a", 0),
        ("validation-a", 5),
    ]


def test_config_records_eager_and_standalone_forward_semantics() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    semantics = config["student"]["forward_semantics"]
    assert semantics["attention_implementation"] == "eager"
    assert semantics["hidden_act"] == "gelu_pytorch_tanh"
    assert semantics["layer_norm_eps"] == 1e-6
    assert semantics["grid"] == [16, 16]
    assert semantics["projected_token_order"] == "camera_major_then_patch_row_major"
    assert config["student"]["evidence_level"] == "standalone_visual_subset"
    assert config["visual_source"]["audit"]["inherited_tensor_count"] == 437
    assert config["visual_source"]["audit"]["backbone_loaded"] is False


def test_selection_uses_retained_best_model_when_historical_full_is_pruned(tmp_path: Path) -> None:
    import hashlib
    import shutil
    from types import SimpleNamespace

    from owac.runtime.raw_full_visual_distillation import _selector_from_validation_history

    root = tmp_path / "run"
    best = root / "best"
    best.mkdir(parents=True)
    model_bytes = b"retained-model-weights"
    (best / "model.pt").write_bytes(model_bytes)
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    old = root / "checkpoints" / "checkpoint_new_000001"
    old.mkdir(parents=True)
    (old / "model.pt").write_bytes(model_bytes)
    latest = root / "checkpoints" / "checkpoint_new_000250"
    latest.mkdir(parents=True)
    (latest / "model.pt").write_bytes(b"latest-model")
    (latest / "optimizer.pt").write_bytes(b"latest-optimizer")
    records = [
        {
            "step": 1,
            "split": "validation",
            "selection_eligible": True,
            "test_used": False,
            "feature_total": 0.5,
            "checkpoint_path": "checkpoints/checkpoint_new_000001",
            "historical_checkpoint_path": "checkpoints/checkpoint_new_000001",
            "retained_model_path": "best/model.pt",
            "checkpoint_sha256": model_sha,
        },
        {
            "step": 250,
            "split": "validation",
            "selection_eligible": True,
            "test_used": False,
            "feature_total": 0.6,
            "checkpoint_path": "checkpoints/checkpoint_new_000250",
            "checkpoint_sha256": "b" * 64,
        },
    ]
    (root / "validation.jsonl").write_text(
        "\n".join(__import__("json").dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    selector = _selector_from_validation_history(SimpleNamespace(run_root=root))
    sidecar = selector.sidecar(updated_at="now")
    assert sidecar["best_step"] == 1
    assert sidecar["checkpoint_path"] == "best/model.pt"
    assert sidecar["checkpoint_sha256"] == model_sha
    # Later validation did not become best: the historical full directory may
    # disappear, but the selected model-only hardlink remains loadable.
    shutil.rmtree(old)
    assert not old.exists()
    assert (root / sidecar["checkpoint_path"]).read_bytes() == model_bytes
    assert (
        hashlib.sha256((root / sidecar["checkpoint_path"]).read_bytes()).hexdigest()
        == sidecar["checkpoint_sha256"]
    )
    assert not (best / "optimizer.pt").exists()
    assert (latest / "optimizer.pt").read_bytes() == b"latest-optimizer"
    assert (latest / "model.pt").read_bytes() == b"latest-model"


def test_map_config_paths_serializes_slots_for_shared_style_aliases(tmp_path: Path) -> None:
    from owac.runtime.raw_full_visual_distillation import _map_config_paths

    physical = tmp_path / "remote" / "repo"
    names = [
        "teacher-registry.json",
        "teacher-frontend.pt",
        "source-manifest.json",
        "load-audit.json",
        "model.safetensors",
        "tokenizer.model",
        "stats.json",
        "cache",
        "author",
        "starvla",
        "configs/pi05.yaml",
    ]
    for name in names:
        path = physical / name
        if name in {"cache", "author", "starvla"}:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
    logical_root = "/home/example/owac/shared-style"
    config = {
        "assets": {
            "teacher_registry": f"{logical_root}/teacher-registry.json",
            "teacher_frontend_checkpoint": f"{logical_root}/teacher-frontend.pt",
            "source_manifest": f"{logical_root}/source-manifest.json",
            "load_audit": f"{logical_root}/load-audit.json",
            "checkpoint": f"{logical_root}/model.safetensors",
            "tokenizer": f"{logical_root}/tokenizer.model",
            "statistics": f"{logical_root}/stats.json",
            "cache_root": f"{logical_root}/cache",
            "author_root": f"{logical_root}/author",
            "starvla_root": f"{logical_root}/starvla",
            "teacher_config_paths": [f"{logical_root}/configs/pi05.yaml"],
        },
        "teacher_requirement": {},
        "remote": {
            "output_root": "/remote/output",
            "path_mapping": [{"logical_prefix": logical_root, "physical_prefix": str(physical)}],
        },
        "data": {"protocol": "owac.shared-rawvla.task6-normal.8-2-2.v1"},
    }
    mapped, report = _map_config_paths(config, PathMapper(config["remote"]["path_mapping"]))
    assert mapped["assets"]["source_manifest"] == str(physical / "source-manifest.json")
    assert mapped["assets"]["cache_root"] == str(physical / "cache")
    assert mapped["assets"]["teacher_config_paths"] == [str(physical / "configs/pi05.yaml")]
    assert len(report["records"]) == 11
    assert all(
        set(item) == {"field", "logical", "physical", "mapped"} for item in report["records"]
    )
    assert all(item["mapped"] is True for item in report["records"])


def test_preflight_receipt_uses_config_digest_and_carries_teacher_prep_wall() -> None:
    from owac.runtime.raw_full_visual_distillation import (
        PREFLIGHT_SCHEMA,
        validate_preflight_receipt,
    )

    old_shape = {
        "schema": PREFLIGHT_SCHEMA,
        "config_digest": "a" * 64,
        "config_sha256": None,
        "cache_identity_key": "b" * 64,
        "optimizer_updates": 0,
        "gpu_wall_sum_seconds": 20.0,
        "finalization_reserve_seconds": 600.0,
    }
    carried = validate_preflight_receipt(
        old_shape,
        config_digest="a" * 64,
        cache_identity_key="b" * 64,
        teacher_preparation_gpu_wall_seconds=100.0,
    )
    assert carried["phase_gpu_wall_seconds"] == 20.0
    assert carried["teacher_preparation_gpu_wall_seconds"] == 100.0
    assert carried["cumulative_gpu_wall_seconds"] == 120.0
    assert carried["gpu_wall_sum_seconds"] == 120.0

    new_shape = {
        **old_shape,
        "preflight_phase_gpu_wall_seconds": 20.0,
        "teacher_preparation_gpu_wall_seconds": 100.0,
        "cumulative_gpu_wall_seconds": 120.0,
        "gpu_wall_sum_seconds": 120.0,
    }
    carried_new = validate_preflight_receipt(
        new_shape,
        config_digest="a" * 64,
        cache_identity_key="b" * 64,
        teacher_preparation_gpu_wall_seconds=100.0,
    )
    assert carried_new["cumulative_gpu_wall_seconds"] == 120.0

    # Legacy digest/cache identity fields are ignored by the active contract.
    ignored = validate_preflight_receipt(
        old_shape,
        config_digest="c" * 64,
        cache_identity_key="c" * 64,
        teacher_preparation_gpu_wall_seconds=100.0,
    )
    assert ignored["cumulative_gpu_wall_seconds"] == 120.0
    with pytest.raises(FullVisualError, match="prep wall"):
        validate_preflight_receipt(
            new_shape,
            config_digest="a" * 64,
            cache_identity_key="b" * 64,
            teacher_preparation_gpu_wall_seconds=90.0,
        )
    with pytest.raises(FullVisualError, match="drops phase"):
        validate_preflight_receipt(
            {**new_shape, "cumulative_gpu_wall_seconds": 110.0},
            config_digest="a" * 64,
            cache_identity_key="b" * 64,
            teacher_preparation_gpu_wall_seconds=100.0,
        )


def test_global_loss_logging_uses_all_unequal_microbatches_not_last() -> None:
    from owac.runtime.raw_full_visual_distillation import (
        finalize_global_loss_components,
        loss_component_numerators,
    )

    # Two ranks, unequal masks and counts.  Values are deliberately unlike the
    # final microbatch so a "last result only" implementation fails.
    rank0 = [
        {"mse": 0.20, "cosine": 0.10, "total": 0.21, "tokens": 100},
        {"mse": 0.50, "cosine": 0.30, "total": 0.53, "tokens": 7},
        {"mse": 0.01, "cosine": 0.90, "total": 0.10, "tokens": 3},
    ]
    rank1 = [
        {"mse": 0.40, "cosine": 0.20, "total": 0.42, "tokens": 11},
        {"mse": 0.05, "cosine": 0.00, "total": 0.05, "tokens": 79},
    ]
    n0 = loss_component_numerators(rank0)
    n1 = loss_component_numerators(rank1)
    combined = {key: n0[key] + n1[key] for key in n0}
    global_tokens = int(combined["tokens"])
    assert global_tokens == 200
    result = finalize_global_loss_components(combined, global_token_total=global_tokens)
    expected_mse = (0.20 * 100 + 0.50 * 7 + 0.01 * 3 + 0.40 * 11 + 0.05 * 79) / 200
    expected_cosine = (0.10 * 100 + 0.30 * 7 + 0.90 * 3 + 0.20 * 11 + 0.00 * 79) / 200
    expected_total = (0.21 * 100 + 0.53 * 7 + 0.10 * 3 + 0.42 * 11 + 0.05 * 79) / 200
    assert result["mse"] == pytest.approx(expected_mse)
    assert result["cosine"] == pytest.approx(expected_cosine)
    assert result["total"] == pytest.approx(expected_total)
    assert result["mse"] != pytest.approx(0.05)  # not rank1's last microbatch
    assert result["scope"] == "global_effective_batch_16"
    with pytest.raises(FullVisualError, match="numerator tokens"):
        finalize_global_loss_components(combined, global_token_total=201)


def test_rng_jsonable_roundtrip_avoids_tensor_object_collectives() -> None:
    import random

    import numpy as np
    import torch

    from owac.runtime.raw_full_visual_distillation import (
        _cpuize,
        _jsonable,
        _restore_rng_state,
    )

    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    np_state = np.random.get_state()
    expected_python = random.getstate()
    expected_numpy = {
        "keys": np.asarray(np_state[1], dtype=np.uint32).copy(),
        "pos": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    expected_torch = torch.get_rng_state().clone()
    state = {
        "python": expected_python,
        "numpy": {
            "kind": str(np_state[0]),
            "keys": torch.from_numpy(expected_numpy["keys"]),
            "pos": expected_numpy["pos"],
            "has_gauss": expected_numpy["has_gauss"],
            "cached_gaussian": expected_numpy["cached_gaussian"],
        },
        "torch": expected_torch,
        "cuda": None,
    }
    encoded = _jsonable(_cpuize(state))
    json.dumps(encoded)  # must be primitive-only for all_gather_object

    def assert_no_tensor(value: object) -> None:
        assert not torch.is_tensor(value)
        if isinstance(value, dict):
            for item in value.values():
                assert_no_tensor(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_tensor(item)

    assert_no_tensor(encoded)
    random.random()
    np.random.random()
    torch.rand(2)
    _restore_rng_state(encoded, "cuda:0")
    assert random.getstate() == expected_python
    restored_numpy = np.random.get_state()
    assert np.array_equal(np.asarray(restored_numpy[1], dtype=np.uint32), expected_numpy["keys"])
    assert int(restored_numpy[2]) == expected_numpy["pos"]
    assert int(restored_numpy[3]) == expected_numpy["has_gauss"]
    assert float(restored_numpy[4]) == expected_numpy["cached_gaussian"]
    assert torch.equal(torch.get_rng_state(), expected_torch)


def test_sampler_state_is_primitive_and_roundtrips_for_object_gather() -> None:
    from owac.runtime.raw_full_visual_distillation import _jsonable
    from owac.training.raw_temporal_distillation import TemporalAnchorSampler

    candidates = [(f"source-{index}", 25 + index * 5) for index in range(20)]
    first = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    encoded = _jsonable(first.state())
    json.dumps(encoded)
    expected = first.next_batch()
    second = TemporalAnchorSampler(candidates, seed=101, batch_size=16)
    second.load_state(encoded)
    assert second.next_batch() == expected


def test_rank1_cuda_rng_is_rank_local_and_legacy_list_is_unambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from owac.runtime.raw_full_visual_distillation import (
        RANK_LOCAL_CUDA_RNG_SCHEMA,
        _capture_rng_state,
        _jsonable,
        _restore_rng_state,
    )

    fake_state = torch.arange(16, dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda index: fake_state.clone())
    captured = _capture_rng_state("cuda:1")
    assert captured["cuda"]["schema"] == RANK_LOCAL_CUDA_RNG_SCHEMA
    assert captured["cuda"]["device_index"] == 1
    encoded = _jsonable(captured)
    json.dumps(encoded)

    calls = []

    def fake_set(state, device=None):
        calls.append((int(device), state.detach().cpu().clone()))

    monkeypatch.setattr(torch.cuda, "set_rng_state", fake_set)
    _restore_rng_state(encoded, "cuda:1")
    assert calls and calls[-1][0] == 1
    assert torch.equal(calls[-1][1], fake_state)

    # Old checkpoints stored exactly one rank-local CUDA state as a list of
    # length one.  It is unambiguous and must restore to cuda:1 without index 1.
    calls.clear()
    _restore_rng_state({"cuda": [fake_state.clone()]}, "cuda:1")
    assert calls and calls[-1][0] == 1
    assert torch.equal(calls[-1][1], fake_state)

    # An explicitly device-0 rank-local state must not be silently applied to
    # rank1's cuda:1.
    with pytest.raises(FullVisualError, match="device 0 != target device 1"):
        _restore_rng_state(
            {
                "cuda": {
                    "schema": RANK_LOCAL_CUDA_RNG_SCHEMA,
                    "device_index": 0,
                    "state": fake_state,
                }
            },
            "cuda:1",
        )

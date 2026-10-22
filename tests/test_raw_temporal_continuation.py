"""CPU tests for the narrow C50->C5000 continuation and checkpoint state."""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from owac.baselines.raw_stem_siglip import sha256_file
from owac.runtime.raw_temporal_continuation import (
    ContinuationBudget,
    _assert_parent_asset_identity,
    _begin_prepare_lifecycle,
    _continuation_record,
    _fail_prepare_lifecycle,
    _normalize_parent_history,
    _prepared_paths,
    owned_run_bytes,
    plan_checkpoint_retention,
    recipe_digest,
    validate_continuation_record,
    validate_migration_manifest,
    validate_parent_checkpoint,
)
from owac.training.raw_temporal_distillation import (
    TemporalAnchorSampler,
    TemporalDistillationTrainer,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/experiments/raw_stem_siglip_c_full_v1.json"
MIGRATION = REPO / "configs/experiments/raw_stem_siglip_c_full_migration_v1.json"
PARENT_CONFIG = REPO / "outputs/raw-distillation-pilot-20261021-101725/authorized-config.json"
PARENT_CHECKPOINT = (
    REPO / "outputs/raw-distillation-pilot-20261021-101725/pilot-C/checkpoint_final_000050.pt"
)
USER_AUTH = REPO / "outputs/raw-distillation-c-full-20261021-123313/user-authorization.json"
AUTHORITY = REPO / "outputs/raw-distillation-c-full-20261021-123313/authority-C.json"
ISO = "2026-10-21T13:00:00+08:00"


def _authorized_manifest(tmp_path: Path, **updates):
    value = json.loads(MIGRATION.read_text(encoding="utf-8"))
    value.update(
        {
            "authorized": True,
            "authorized_by": "cpu-test",
            "authorized_at": ISO,
        }
    )
    value.update(updates)
    target = tmp_path / "migration.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    return target


def _validate(manifest: Path):
    return validate_migration_manifest(
        manifest,
        new_config_path=CONFIG,
        parent_config_path=PARENT_CONFIG,
        parent_checkpoint_path=PARENT_CHECKPOINT,
        user_authorization_path=USER_AUTH,
    )


def _parent_payload(
    recipe: str = "ccd0ec412f85113e28f54f4dc50dab41ae5f6d39e858ef3e8988290d404b7477",
):
    return {
        "schema": "owac.raw-stem-temporal.checkpoint.v1",
        "arm": "C",
        "scope": "stem_plus_gate",
        "cumulative_updates": 50,
        "anchor_presentations": 800,
        "run_identity": {
            "research_config_sha256": recipe,
            "scope": "stem_plus_gate",
            "cache_identity_key": "a" * 64,
        },
        "training_config": {
            "max_cumulative_updates_per_arm": 5000,
            "pilot_max_updates_per_arm": 50,
        },
        "optimizer": {"state": {"0": {"step": 50}, "1": {"step": 50}}},
        "scheduler": {"last_epoch": 50},
        "sampler": {"cursor": 7},
        "rng": {"torch": "state"},
        "model_identity": "b" * 64,
    }


def _continuation_payload(tmp_path: Path, *, authority_sha: str = "c" * 64):
    migration = _validate(_authorized_manifest(tmp_path))
    record = _continuation_record(
        migration=migration,
        authority_sha256=authority_sha,
        interval=250,
        milestones=(500, 2000, 5000),
        wall_cap=230400,
        output_cap=21474836480,
    )
    payload = {
        "schema": "owac.raw-stem-temporal.checkpoint.v1",
        "arm": "C",
        "scope": "stem_plus_gate",
        "cumulative_updates": 51,
        "anchor_presentations": 816,
        "optimizer": {"state": {"0": {"step": 51}}},
        "scheduler": {"last_epoch": 51},
        "sampler": {"batch_size": 16, "cursor": 3},
        "rng": {"torch": "state"},
        "run_identity": {"research_config_sha256": migration.new_recipe_sha256},
        "training_config": {"pilot_max_updates_per_arm": 5000},
        "continuation": record,
    }
    expected = {
        "new_recipe_sha256": migration.new_recipe_sha256,
        "old_recipe_sha256": migration.old_recipe_sha256,
        "parent_checkpoint_sha256": migration.parent_checkpoint_sha256,
        "parent_config_sha256": migration.parent_config_sha256,
        "migration_sha256": migration.manifest_sha256,
        "authority_sha256": authority_sha,
    }
    return payload, expected, migration


def test_recipe_digest_matches_existing_runtime_helper() -> None:
    from owac.runtime.raw_temporal_distillation import _research_config_digest

    for path in (PARENT_CONFIG, CONFIG):
        value = json.loads(path.read_text(encoding="utf-8"))
        assert recipe_digest(value) == _research_config_digest(value)
    assert recipe_digest(json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))) == (
        "ccd0ec412f85113e28f54f4dc50dab41ae5f6d39e858ef3e8988290d404b7477"
    )


def test_migration_accepts_exact_narrow_operational_change(tmp_path: Path) -> None:
    result = _validate(_authorized_manifest(tmp_path))
    budget = ContinuationBudget()
    assert result.old_recipe_sha256 != result.new_recipe_sha256
    assert result.run_root.name == "raw-distillation-c-full-20261021-123313"
    assert {entry["kind"] for entry in result.carried_receipts} == {"parity", "resource"}
    assert all(entry["rewritten"] is False for entry in result.carried_receipts)
    assert result.expected_parent_baseline["feature_mse"] == pytest.approx(2.7655930869719563)
    assert result.expected_parent_baseline["feature_cosine"] == pytest.approx(0.047423013905976336)
    assert result.expected_parent_baseline["valid_tokens"] == 52224
    assert result.expected_parent_baseline["scalar_denominator"] == 106954752.0
    assert budget.max_new_updates == 4950 and budget.max_new_anchors == 79200


def test_migration_template_is_not_authorized() -> None:
    with pytest.raises(ValueError, match="not authorized"):
        _validate(MIGRATION)


def test_migration_rejects_arm_unfreeze_and_sha_mismatches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C-only"):
        _validate(_authorized_manifest(tmp_path, arm="B"))
    with pytest.raises(ValueError, match="vision_unfreeze"):
        _validate(_authorized_manifest(tmp_path, vision_unfreeze=True))
    with pytest.raises(ValueError, match="parent_checkpoint_sha256"):
        _validate(_authorized_manifest(tmp_path, parent_checkpoint_sha256="0" * 64))
    with pytest.raises(ValueError, match="old_recipe_sha256"):
        _validate(_authorized_manifest(tmp_path, old_recipe_sha256="1" * 64))
    with pytest.raises(ValueError, match="declared_caps"):
        _validate(_authorized_manifest(tmp_path, declared_caps={"max_cumulative_updates": 6000}))
    with pytest.raises(ValueError, match="authorized_by"):
        _validate(_authorized_manifest(tmp_path, authorized_by=""))


def _mutated_new_config(tmp_path: Path, mutate) -> Path:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    mutate(value)
    target = tmp_path / "mutated-config.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    return target


def _manifest_for_new_config(tmp_path: Path, new_config: Path) -> Path:
    manifest = json.loads(MIGRATION.read_text(encoding="utf-8"))
    manifest.update(
        {
            "authorized": True,
            "authorized_by": "cpu-test",
            "authorized_at": ISO,
            "new_config": str(new_config),
            "new_config_sha256": sha256_file(new_config),
            "new_recipe_sha256": recipe_digest(json.loads(new_config.read_text())),
        }
    )
    target = tmp_path / "mutated-migration.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["training"].__setitem__("learning_rate", 2e-4),
        lambda value: value["training"].__setitem__("seed", 999),
        lambda value: value["training"].__setitem__("optimizer", "Adam"),
        lambda value: value["student"].__setitem__("stem_seed", 202),
        lambda value: value["assets"].__setitem__("cache_root", "/tmp/other-cache"),
        lambda value: value["training"].__setitem__("max_cumulative_updates_per_arm", 4000),
        lambda value: value["evaluation"].__setitem__("test_selection", True),
        lambda value: value["task"].__setitem__("suite", "libero_spatial"),
    ],
)
def test_migration_rejects_identity_changes(tmp_path: Path, mutate) -> None:
    new_config = _mutated_new_config(tmp_path, mutate)
    manifest = _manifest_for_new_config(tmp_path, new_config)
    with pytest.raises(ValueError):
        validate_migration_manifest(
            manifest,
            new_config_path=new_config,
            parent_config_path=PARENT_CONFIG,
            parent_checkpoint_path=PARENT_CHECKPOINT,
            user_authorization_path=USER_AUTH,
        )


def test_parent_checkpoint_accepts_actual_identity_and_rejects_drift() -> None:
    good = _parent_payload()
    audit = validate_parent_checkpoint(
        good,
        parent_checkpoint_sha256="d" * 64,
        old_recipe_sha256="ccd0ec412f85113e28f54f4dc50dab41ae5f6d39e858ef3e8988290d404b7477",
        parent_config_sha256="e" * 64,
    )
    assert audit["next_optimizer_step"] == 51
    assert audit["optimizer_step_counters"] == [50]
    for mutate in (
        lambda value: value.__setitem__("cumulative_updates", 49),
        lambda value: value.__setitem__("anchor_presentations", 799),
        lambda value: value.__setitem__("arm", "B"),
        lambda value: value["optimizer"]["state"]["0"].__setitem__("step", 49),
        lambda value: value["scheduler"].__setitem__("last_epoch", 49),
        lambda value: value["run_identity"].__setitem__("research_config_sha256", "f" * 64),
        lambda value: value.pop("rng"),
    ):
        payload = copy.deepcopy(good)
        mutate(payload)
        with pytest.raises(ValueError):
            validate_parent_checkpoint(
                payload,
                parent_checkpoint_sha256="d" * 64,
                old_recipe_sha256="ccd0ec412f85113e28f54f4dc50dab41ae5f6d39e858ef3e8988290d404b7477",
                parent_config_sha256="e" * 64,
            )


def test_continuation_record_accepts_and_rejects_state_drift(tmp_path: Path) -> None:
    payload, expected, _migration = _continuation_payload(tmp_path)
    audit = validate_continuation_record(payload, expected=expected)
    assert audit["next_optimizer_step"] == 52
    for mutate in (
        lambda value: value.__setitem__("cumulative_updates", 50),
        lambda value: value.__setitem__("anchor_presentations", 815),
        lambda value: value["optimizer"]["state"]["0"].__setitem__("step", 50),
        lambda value: value["scheduler"].__setitem__("last_epoch", 50),
        lambda value: value["run_identity"].__setitem__("research_config_sha256", "0" * 64),
        lambda value: value["continuation"].__setitem__("vision_unfreeze", True),
        lambda value: value["continuation"].__setitem__("new_stage_elapsed_excludes_parent", False),
        lambda value: value["continuation"].__setitem__("milestones", [500, 2000]),
        lambda value: value.pop("rng"),
    ):
        broken = copy.deepcopy(payload)
        mutate(broken)
        with pytest.raises(ValueError):
            validate_continuation_record(broken, expected=expected)


def test_checkpoint_retention_keeps_recovery_milestones_best_and_final() -> None:
    entries = [
        {"name": "checkpoint_000100.pt"},
        {"name": "checkpoint_000250.pt"},
        {"name": "checkpoint_000500.pt"},
        {"name": "checkpoint_000750.pt"},
        {"name": "checkpoint_002000.pt"},
        {"name": "checkpoint_005000.pt"},
        {"name": "best.pt"},
        {"name": "latest.pt"},
        {"name": "checkpoint_final_0005000.pt"},
    ]
    plan = plan_checkpoint_retention(entries, current_step=5000, best_step=250, final_step=5000)
    assert plan["keep"] == [
        "checkpoint_000250.pt",
        "checkpoint_000500.pt",
        "checkpoint_002000.pt",
        "checkpoint_005000.pt",
    ]
    assert plan["delete"] == ["checkpoint_000100.pt", "checkpoint_000750.pt"]
    assert "best.pt" not in plan["delete"]
    assert "checkpoint_final_0005000.pt" not in plan["delete"]


def test_owned_run_bytes_skips_symlinks_and_dedupes_hardlinks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "a.bin").write_bytes(b"1234567890")
    os.link(root / "a.bin", root / "hardlink.bin")
    (root / "harness").mkdir()
    (root / "harness" / "log.txt").write_bytes(b"log")
    os.symlink("/etc/passwd", root / "cache-link")
    os.symlink(root / "a.bin", root / "file-link")
    assert owned_run_bytes(root) == 13


def test_continuation_budget_matches_user_authorization() -> None:
    from owac.runtime.raw_temporal_continuation import budget_from_user_authorization

    budget = budget_from_user_authorization(USER_AUTH)
    assert budget == ContinuationBudget()
    assert budget.max_new_updates == 4950
    assert budget.max_new_anchors == 79200


def test_rng_state_roundtrip_is_exact() -> None:
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    state = TemporalDistillationTrainer.capture_rng_state()
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    TemporalDistillationTrainer.restore_rng_state(state)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_anchor_sampler_roundtrip_preserves_next_batches() -> None:
    candidates = [(f"source/{index}", 25 + 5 * index) for index in range(20)]
    sampler = TemporalAnchorSampler(candidates, seed=101, batch_size=4)
    for _ in range(5):
        sampler.next_batch()
    state = sampler.state()
    expected = [sampler.next_batch() for _ in range(5)]
    restored = TemporalAnchorSampler(candidates, seed=101, batch_size=4)
    restored.load_state(state)
    actual = [restored.next_batch() for _ in range(5)]
    assert actual == expected
    assert "candidate_fingerprint" not in state
    assert state["candidates"] == restored.candidates
    assert state["seed"] == 101 and state["batch_size"] == 4


def test_optimizer_and_scheduler_roundtrip_preserves_next_update() -> None:
    torch.manual_seed(11)
    features = torch.randn(8, 4)
    targets = torch.randn(8, 1)
    model = nn.Linear(4, 1)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / 100))
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(features) - targets).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    assert scheduler.state_dict()["last_epoch"] == 50
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5.1e-5)
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict())

    restored_model = nn.Linear(4, 1)
    restored_model.load_state_dict(model_state)
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-4, weight_decay=0.01)
    restored_scheduler = LambdaLR(
        restored_optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / 100)
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    assert restored_scheduler.state_dict()["last_epoch"] == 50
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(5.1e-5)

    for current, current_optimizer, current_scheduler in (
        (model, optimizer, scheduler),
        (restored_model, restored_optimizer, restored_scheduler),
    ):
        current_optimizer.zero_grad(set_to_none=True)
        loss = (current(features) - targets).square().mean()
        loss.backward()
        current_optimizer.step()
        current_scheduler.step()
    for name, tensor in model.state_dict().items():
        assert torch.allclose(tensor, restored_model.state_dict()[name], atol=0.0, rtol=0.0)
    assert restored_scheduler.state_dict()["last_epoch"] == 51
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(5.2e-5)
    steps = {
        int(value["step"])
        for value in restored_optimizer.state_dict()["state"].values()
        if "step" in value
    }
    assert steps == {51}


def test_parent_history_normalization_ignores_torn_tail(tmp_path: Path) -> None:
    source = tmp_path / "progress.jsonl"
    lines = []
    for step in range(1, 51):
        lines.append(
            json.dumps(
                {
                    "event": "update",
                    "cumulative_updates": step,
                    "anchor_presentations": step * 16,
                    "feature_mse": 1.0 + step,
                    "feature_cosine": 0.1,
                    "feature_total": 1.1 + step,
                    "learning_rate": 1e-5,
                    "gpu_allocated_bytes": 1,
                    "gpu_reserved_bytes": 2,
                }
            )
        )
    source.write_text(
        "\n".join(lines) + '\n{"event":"update","cumulative_updates":51', encoding="utf-8"
    )
    target = tmp_path / "parent-history.jsonl"
    records = _normalize_parent_history(source, target)
    assert len(records) == 50
    assert records[0]["cumulative_updates"] == 1
    assert records[-1]["cumulative_updates"] == 50
    assert records[-1]["anchor_presentations"] == 800
    assert len(target.read_text(encoding="utf-8").splitlines()) == 50

    source.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="50 updates"):
        _normalize_parent_history(source, target)


def test_authority_receipt_binds_current_user_authorization() -> None:
    from owac.runtime.raw_temporal_continuation import _authority_payload

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    value = _authority_payload(AUTHORITY, config=config, user_authorization=USER_AUTH)
    assert value["max_updates"] == 5000
    assert value["max_wall_seconds"] == 230400
    assert value["max_output_bytes"] == 21474836480
    assert value["receipt_sha256"] == sha256_file(USER_AUTH)


def test_parent_asset_identity_uses_actual_teacher_artifact_key() -> None:
    bundle = {
        "artifact_sha256": "a" * 64,
        "backend_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "stats_sha256": "d" * 64,
        "sources_sha256": "e" * 64,
    }
    # Actual C50 run_identity key shape.
    parent = {
        "teacher_artifact_sha256": "a" * 64,
        "backend_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "stats_sha256": "d" * 64,
        "sources_sha256": "e" * 64,
    }
    _assert_parent_asset_identity(parent, bundle)

    wrong_key = dict(parent)
    wrong_key["artifact_sha256"] = wrong_key.pop("teacher_artifact_sha256")
    with pytest.raises(ValueError, match="teacher_artifact_sha256"):
        _assert_parent_asset_identity(wrong_key, bundle)

    mismatch = dict(parent)
    mismatch["teacher_artifact_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="teacher_artifact_sha256"):
        _assert_parent_asset_identity(mismatch, bundle)

    missing = dict(parent)
    missing.pop("sources_sha256")
    with pytest.raises(ValueError, match="sources_sha256"):
        _assert_parent_asset_identity(missing, bundle)


def test_prepare_lifecycle_publishes_pid_and_durable_error(tmp_path: Path) -> None:
    from types import SimpleNamespace

    root = tmp_path / "run"
    root.mkdir()
    paths = _prepared_paths(root)
    migration = SimpleNamespace(
        new_recipe_sha256="a" * 64,
        parent_checkpoint_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )
    _status, _heartbeat = _begin_prepare_lifecycle(
        paths,
        migration=migration,
        authority_sha="d" * 64,
        target_cumulative=5000,
    )
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    heartbeat = json.loads(paths["heartbeat"].read_text(encoding="utf-8"))
    assert status["status"] == "preparing"
    assert status["phase"] == "preparing"
    assert status["pid"] == os.getpid()
    assert status["pid_start_ticks"] is not None
    assert status["cumulative_updates"] == 50
    assert heartbeat["schema"].endswith("heartbeat.v1")
    assert heartbeat["pid"] == os.getpid()
    assert heartbeat["pid_start_ticks"] == status["pid_start_ticks"]
    assert heartbeat["phase"] == "preparing"

    _fail_prepare_lifecycle(ValueError("fixture pre-asset failure"))
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    assert status["status"] == "error"
    assert status["phase"] == "error"
    assert status["native_exit"] == 2
    assert status["prepare_wall_seconds"] >= 0.0
    assert "fixture pre-asset failure" in status["error"]

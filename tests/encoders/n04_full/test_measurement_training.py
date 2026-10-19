"""S1 训练、恢复、导出与评测的合同测试（合成数据，不依赖远端数据集）。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.evaluation.n04_full import carry_diagnostics, evaluate_main, protocol_comparison
from owac.training.n04_full import (
    SequenceStore,
    Trainer,
    build_index,
    build_scene_views,
    build_step_batch,
    data_stats_main,
    export_main,
    geometry_loss,
    lr_grid_main,
    profile_main,
    sample_plan,
)


@pytest.fixture(scope="module")
def store(s1_config) -> SequenceStore:
    return SequenceStore(max_frames_per_sequence=s1_config.data.max_frames_per_sequence)


def test_index_groups_by_platform_and_shape(s1_config, store):
    groups = build_index(s1_config, ("train", "dev"), store)
    sizes = {f"{key[0]}|{key[1]}": len(refs) for key, refs in groups.items()}
    assert sizes["libero|(64, 48)"] == 3 + 2  # train 3 帧 + dev 2 帧（单配对）
    assert sizes["robotwin2|(48, 64)"] == 6  # 3 帧 × 2 配对
    assert sizes["rendered_factor|(32, 32)"] == 3 * 2 + 3 * 2  # 每源两个 factor 变体
    train_only = build_index(s1_config, ("train",), store)
    assert {f"{k[0]}" for k in train_only} == {"libero", "robotwin2", "rendered_factor"}


def test_sample_plan_is_deterministic_and_homogeneous(s1_config, store):
    groups = build_index(s1_config, ("train",), store)
    plan = sample_plan(groups, seed=17, step=3, batch_frames=8, micro_batch_frames=4)
    assert len(plan) == 2 and all(len(chunk) == 2 for chunk in plan)
    again = sample_plan(groups, seed=17, step=3, batch_frames=8, micro_batch_frames=4)
    assert [[ref.measurement_key for ref in chunk] for chunk in plan] == [
        [ref.measurement_key for ref in chunk] for chunk in again
    ]
    other_step = sample_plan(groups, seed=17, step=4, batch_frames=8, micro_batch_frames=4)
    assert [[ref.measurement_key for ref in chunk] for chunk in plan] != [
        [ref.measurement_key for ref in chunk] for chunk in other_step
    ]
    for chunk in plan:
        assert len({ref.shape_key for ref in chunk}) == 1


def test_scene_views_use_only_online_inputs(s1_config, store):
    groups = build_index(s1_config, ("train",), store)
    ref = groups[("libero", (64, 48))][0]
    frame_a, frame_b, target_a, target_b = build_scene_views(ref, store)
    arrays = store.paired_frame(ref.path, ref.frame_index)
    expected = torch.from_numpy(np.ascontiguousarray(arrays["online_rawvla_rgb3"][0])).permute(
        2, 0, 1
    )
    assert torch.equal(frame_a.raw, expected)
    assert frame_a.K is not None and frame_b.K is not None
    assert torch.equal(frame_a.K, torch.from_numpy(arrays["online_K"][0]).to(torch.float32))
    assert target_a.depth.shape == ref.shape_hw
    assert not bool(target_a.support[0, 5])  # 前 4 行是无效深度
    assert bool(target_a.support[10, 5])
    # 目标里不得出现任何 online_* 或未来字段
    assert not hasattr(target_a, "raw")
    assert set(vars(frame_a)) <= {
        "raw",
        "camera_id",
        "measurement_id",
        "input_mode",
        "scale",
        "size_hw",
        "K",
        "cfa_pattern",
        "time_s",
        "time_unit",
        "saturation_level",
        "invalid_mask",
        "provenance",
    }


def test_geometry_loss_is_finite_and_masks_invalid(s1_config, store):
    trainer = Trainer(s1_config, seed=17, run_dir="/tmp/n04_s1_geometry_test")
    refs = build_index(s1_config, ("train",), store)[("libero", (64, 48))][:1]
    batch = build_step_batch(refs, store)
    out = trainer.model.measure(batch.frames_a)
    loss, stats = geometry_loss(
        depth_pred=out.depth_mean,
        variance_pred=out.depth_variance,
        targets=batch.targets_a,
        grids=[out.sample_grid],
        config=s1_config,
    )
    assert torch.isfinite(loss) and loss.item() > 0
    # 无效行（前 4 行原图）覆盖的融合格必须被排除：有效点数严格小于全部格数
    assert 0 < stats["depth_points"].item() < float(out.depth_mean[0, 0].numel())


def test_correspondence_loss_finds_known_projected_pairs(s1_config, store):
    trainer = Trainer(s1_config, seed=17, run_dir="/tmp/n04_s1_corr_test")
    refs = build_index(s1_config, ("train",), store)[("libero", (64, 48))][:1]
    batch = build_step_batch(refs, store)
    out_a = trainer.model.measure(batch.frames_a)
    out_b = trainer.model.measure(batch.frames_b)
    from owac.training.n04_full import correspondence_loss

    loss, stats = correspondence_loss(
        out_a=out_a,
        out_b=out_b,
        targets_a=batch.targets_a,
        targets_b=batch.targets_b,
        config=s1_config,
    )
    assert torch.isfinite(loss)
    # 合成几何：深度 1.0、基线 0.1、fx 50 -> 位移 5 px，有效对应点必须存在
    assert stats["pairs"].item() > 0
    assert "positive_cosine" in stats and "top1" in stats


def test_training_step_updates_parameters_and_reaches_every_stage(s1_config, store):
    config = replace(
        s1_config, training=replace(s1_config.training, batch_frames=4, micro_batch_frames=4)
    )
    trainer = Trainer(config, seed=17, run_dir="/tmp/n04_s1_step_test")
    trainer.train_groups = build_index(config, ("train",), store)
    before = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    info = trainer.train_step(0)
    assert all(np.isfinite(value) for key, value in info.items() if key != "count")
    groups = {}
    for name, parameter in trainer.model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        parts = name.split(".")
        if name.startswith(("backbone.stages", "backbone.downsample_layers")):
            group = ".".join(parts[:3])
        else:
            group = parts[0]
        groups[group] = groups.get(group, 0.0) + float(parameter.grad.abs().max())
    for expected in (
        "backbone.stages.0",
        "backbone.stages.1",
        "backbone.stages.2",
        "backbone.stages.3",
        "backbone.downsample_layers.0",
        "backbone.downsample_layers.3",
        "laterals",
        "fusion",
        "depth_head",
        "variance_head",
    ):
        assert groups.get(expected, 0.0) > 0.0, expected
    changed = [
        name
        for name, parameter in trainer.model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]
    assert changed, "一步训练必须更新参数"
    assert info["lr"] > 0


def test_optimizer_groups_cover_all_trainable_parameters(s1_config):
    trainer = Trainer(s1_config, seed=17, run_dir="/tmp/n04_s1_opt_test")
    seen = []
    for group in trainer.optimizer.param_groups:
        seen.extend(id(parameter) for parameter in group["params"])
    model_params = [
        id(parameter) for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    assert sorted(seen) == sorted(model_params)
    assert len(seen) == len(set(seen))
    assert trainer.param_groups_report["decay_params"] + trainer.param_groups_report[
        "no_decay_params"
    ] == sum(parameter.numel() for parameter in trainer.model.parameters())


def test_train_run_and_resume_reproduce_step(s1_config, store, tmp_path):
    config = replace(s1_config, training=replace(s1_config.training, batch_frames=4, steps=2))
    run_a = Trainer(config, seed=17, run_dir=tmp_path / "a")
    run_a.train_groups = build_index(config, ("train",), store)
    losses = [run_a.train_step(step)["total"] for step in range(2)]
    checkpoint = tmp_path / "a" / "checkpoint.pt"
    run_a.save_checkpoint(checkpoint)

    run_b = Trainer(config, seed=17, run_dir=tmp_path / "b")
    run_b.train_groups = build_index(config, ("train",), store)
    payload = run_b.load_checkpoint(checkpoint)
    assert payload["step"] == 2
    assert run_b.scheduler.get_last_lr() == run_a.scheduler.get_last_lr()
    resumed_loss = run_b.train_step(2)["total"]
    reference = Trainer(config, seed=17, run_dir=tmp_path / "c")
    reference.train_groups = build_index(config, ("train",), store)
    for step in range(2):
        reference.train_step(step)
    continued = reference.train_step(2)["total"]
    assert resumed_loss == pytest.approx(continued, rel=1e-5, abs=1e-6)
    assert losses[0] != losses[1]


def test_checkpoint_rejects_wrong_identity(s1_config, tmp_path):
    trainer = Trainer(s1_config, seed=17, run_dir=tmp_path / "x")
    path = tmp_path / "x" / "checkpoint.pt"
    trainer.save_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["schema"] = "n04-full-v0"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        trainer.load_checkpoint(path)


def test_profile_and_data_stats_and_lr_grid(s1_config, store, tmp_path):
    profile = profile_main(s1_config, tmp_path / "profile", steps=2)
    assert profile["steps"] == 2 and profile["params"] == 88_942_210
    assert profile["gradient_audit"]["params_without_grad"] == []
    assert profile["gradient_audit"]["params_nonfinite_grad"] == []
    stats = data_stats_main(s1_config, tmp_path / "stats")
    median = stats["splits"]["train"]["depth_median_m"]
    assert median == pytest.approx(1.0, abs=0.2)
    grid = lr_grid_main(s1_config, tmp_path / "lr", steps=2)
    assert len(grid["results"]) == len(s1_config.training.lr_grid)


def test_evaluate_protocol_and_carry(s1_config, store, tmp_path):
    config = replace(s1_config, training=replace(s1_config.training, batch_frames=4))
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "eval")
    path = tmp_path / "eval" / "checkpoint.pt"
    trainer.save_checkpoint(path)
    report = evaluate_main(
        config,
        str(path),
        split="dev",
        output=tmp_path / "dev.json",
        protocol=None,
        with_carry=True,
    )
    metrics = report["metrics"]
    for key in (
        "depth_mae_m",
        "coverage_1sigma",
        "coverage_2sigma",
        "variance_std_m",
        "position_error_m",
    ):
        assert key in metrics
    assert metrics["groups"], "必须给出分组指标"
    assert report["checkpoint_sha256"]
    assert set(report["memoryless_carry"]["latent_relative_drift"]) <= {"1", "2", "3"}
    protocol = protocol_comparison(
        Path(__file__).resolve().parents[3] / "configs/n04/full/protocol_calibrated_sim.json",
        metrics,
    )
    assert protocol["available"] is True
    assert protocol["groups"], "标定协议里必须能对上至少一个分组"
    assert protocol["groups"]["libero/agentview"]["protocol_ratio"] > 0


def test_export_is_deterministic_and_verifies(s1_config, store, tmp_path):
    config = replace(s1_config, training=replace(s1_config.training, batch_frames=4))
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "train")
    checkpoint = tmp_path / "train" / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    first = export_main(
        config, tmp_path / "cache_a", str(checkpoint), splits=("dev",), verify_samples=2
    )
    second = export_main(
        config, tmp_path / "cache_b", str(checkpoint), splits=("dev",), verify_samples=2
    )
    assert first["entries"] == second["entries"] > 0
    index_a = json.loads((tmp_path / "cache_a" / "index.json").read_text())
    index_b = json.loads((tmp_path / "cache_b" / "index.json").read_text())
    hashes_a = [
        (entry["measurement_id"], entry["measurement_sha256"]) for entry in index_a["entries"]
    ]
    hashes_b = [
        (entry["measurement_id"], entry["measurement_sha256"]) for entry in index_b["entries"]
    ]
    assert hashes_a == hashes_b
    verify = json.loads((tmp_path / "cache_a" / "verify.json").read_text())
    assert verify["samples"] > 0
    for result in verify["results"]:
        assert result["entries_match"] is True
        assert max(result["max_abs_diff"].values()) == 0.0
    # GT 与测量分开文件，缓存里不得混入 gt_*
    cached = np.load(
        tmp_path / "cache_a" / index_a["entries"][0]["measurement_path"], allow_pickle=False
    )
    assert not any(name.startswith("gt_") for name in cached.files)
    gt = np.load(tmp_path / "cache_a" / index_a["entries"][0]["gt_path"], allow_pickle=False)
    assert set(gt.files) == {"depth", "support", "K", "T_W_from_C"}


def test_carry_diagnostics_reports_age_trend(s1_config, store, tmp_path):
    config = replace(s1_config, training=replace(s1_config.training, batch_frames=4))
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "carry")
    diagnostics = carry_diagnostics(config, trainer, "dev", max_age=2)
    assert diagnostics["sequences"] >= 1
    assert diagnostics["latent_relative_drift"]


def test_factor_variants_get_distinct_measurement_ids(s1_config, store):
    """同一源下不同 factor 文件必须有不同的 measurement ID（否则缓存互相覆盖）。"""

    groups = build_index(s1_config, ("train",), store)
    factor_refs = groups[("rendered_factor", (32, 32))]
    variants = {ref.variant for ref in factor_refs}
    assert len(variants) >= 2, "夹具必须包含同一 seed 的多个 factor 变体"
    ids = {}
    for ref in factor_refs:
        frame, _, _, _ = build_scene_views(ref, store)
        ids.setdefault(ref.variant, set()).add(frame.measurement_id)
    assert all(len(value) == len(factor_refs) // len(variants) for value in ids.values())
    union = set().union(*ids.values())
    assert len(union) == sum(len(value) for value in ids.values()), (
        "变体之间不得共享 measurement ID"
    )


def test_export_entries_are_unique_per_variant(s1_config, store, tmp_path):
    config = replace(s1_config, training=replace(s1_config.training, batch_frames=4))
    trainer = Trainer(config, seed=17, run_dir=tmp_path / "train")
    checkpoint = tmp_path / "train" / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    export_main(config, tmp_path / "cache", str(checkpoint), splits=("dev",), verify_samples=6)
    index = json.loads((tmp_path / "cache" / "index.json").read_text())
    ids = [entry["measurement_id"] for entry in index["entries"]]
    paths = [entry["measurement_path"] for entry in index["entries"]]
    keys = [
        (entry["source_path"], entry["frame_index"], entry["camera_id"])
        for entry in index["entries"]
    ]
    assert len(set(ids)) == len(ids), "measurement ID 必须唯一"
    assert len(set(paths)) == len(paths), "缓存文件不得互相覆盖"
    assert len(set(keys)) == len(keys)
    verify = json.loads((tmp_path / "cache" / "verify.json").read_text())
    assert verify["samples"] == 6
    for result in verify["results"]:
        assert result["entries_match"] is True
        assert max(result["max_abs_diff"].values()) == 0.0


def test_detect_stability_uses_recent_window_only(s1_config):
    """趋稳判据只看最近窗口：早期大幅改善不能让"已平台"永远报 still_improving。"""
    from owac.training.n04_full import detect_stability

    early = [{"depth_norm_mae": value} for value in (1.0, 0.5, 0.25, 0.12, 0.06, 0.03)]
    flat = [{"depth_norm_mae": value} for value in (0.03, 0.0299, 0.0301, 0.0298, 0.0302, 0.0300)]
    verdict_early = detect_stability(early, s1_config)
    verdict_flat = detect_stability(flat, s1_config)
    assert verdict_early["converged"] is False and verdict_early["reason"] == "still_improving"
    assert verdict_flat["converged"] is True
    assert len(verdict_flat["last_improvements"]) == s1_config.training.stability_window
    assert detect_stability(flat[:3], s1_config)["reason"] == "insufficient_evals"

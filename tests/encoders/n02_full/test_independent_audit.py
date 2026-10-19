"""Regressions found by comparing P0 requirements to real source metadata and execution."""

import dataclasses
import json

import numpy as np
import pytest

from owac.evaluation.n02_full import p0
from owac.frameworks.n02_full import data as d
from owac.frameworks.n02_full.config import N02FullConfig
from owac.raw.rawvla_sequence_n02 import RawVLAWindowDataset, WindowOptions


def test_task_stratification_and_declared_source_identity(tmp_path, libero_writer):
    root = tmp_path / "cache"
    for task in ("task_a", "task_b"):
        for episode in range(20):
            # Distinct path identities, same declared demo: must not leak across splits.
            for variant in range(2):
                metadata = dict(
                    task_name=task,
                    suite="libero_suite",
                    hdf5_path=task + ".hdf5",
                    demo_key=f"demo_{episode}",
                )
                libero_writer(
                    root,
                    "suite_directory",
                    episode + (100 if task == "task_b" else 0),
                    variant,
                    extra={"metadata_json": np.array(json.dumps(metadata))},
                )
    loader = RawVLAWindowDataset(root, dataset_format="libero")
    manifest = d.build_partition_manifest(
        loader.episodes(), dataset_format="libero", cache_root=root, seed=1103
    )
    assert d.verify_partitions(manifest)["violations"] == []
    for task in ("task_a", "task_b"):
        counts = {
            part: sum(manifest.entry_of(path).task_id == task for path in paths)
            for part, paths in manifest.partitions.items()
        }
        assert counts == dict(train=32, dev=4, calibration=4)
    assert len(manifest.groups) == 40
    assert all(".hdf5/demo_" in e.source_trajectory_id for e in manifest.entries)
    assert all("scene_id" in e.missing_fields for e in manifest.entries)


def test_anchor_window_is_consumed_and_conflicts_rejected(tmp_path, libero_writer):
    for episode in range(6):
        libero_writer(tmp_path, "suite", episode, 0, num_frames=24)
    loader = RawVLAWindowDataset(tmp_path, dataset_format="libero")
    manifest = d.build_partition_manifest(
        loader.episodes(), dataset_format="libero", cache_root=tmp_path, seed=1
    )
    table = d.build_anchor_table(manifest, "train", stride=2, rnn_frames=3, action_horizon=4)
    dataset = d.ManifestWindowDataset(tmp_path, manifest, "train", table)
    sample = dataset[3]
    assert sample.supervision_actions.shape[0] == 4
    assert tuple(sample.observation_indices) == table[3].history_indices
    with pytest.raises(ValueError, match="frozen anchor"):
        d.ManifestWindowDataset(
            tmp_path, manifest, "train", table, window_options=WindowOptions(action_horizon=10)
        )


def test_profile_config_consumes_dataclass():
    from owac.frameworks.n02_full.components.profile_targets import _profile_config

    cfg = dataclasses.replace(N02FullConfig(), profile={"batch_size": 3})
    assert _profile_config(cfg)["batch_size"] == 3


def test_profile_training_step_changes_parameters():
    import torch

    model = torch.nn.Linear(4, 2)
    model.profile_meta = dict(input_shape=(2, 4), batch_size=2, operation="train_step", params=10)
    before = model.weight.detach().clone()
    measured, failure = p0._measure_target(
        p0._ProfileTarget(model, 0, "linear"), "cpu", 3, 1, {"cuda_queried": False}
    )
    assert failure is None
    assert measured["optimizer_steps"] == 4
    assert not torch.equal(before, model.weight)
    assert measured["operation"] == "train_step"


def test_missing_stage_cannot_pass():
    decision = p0.build_stage_decision({}, [])
    assert decision["implementation"] != "pass"


def test_sparse_block_preserves_coordinates_and_matches_gathered_feature_rows():
    import copy

    import torch
    import torch.nn.functional as f

    from owac.frameworks.n02_full.components.fcmae import MaskAwareBlock

    torch.manual_seed(9)
    block = MaskAwareBlock(3).double()
    with torch.no_grad():
        block.grn.gamma.fill_(0.7)
        block.grn.beta.fill_(0.2)
    reference = copy.deepcopy(block)
    valid = torch.zeros(2, 1, 9, 9, dtype=torch.float64)
    valid[0, 0, 2, 2] = valid[0, 0, 3, 4] = valid[1, 0, 5, 5] = 1
    x = torch.randn(2, 3, 9, 9, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_()
    got, support = block(x, valid)
    active = valid[:, 0].bool()
    # Independent sparse representation: three coordinate rows, no absent sites.
    dense = f.conv2d(
        x_ref * valid, reference.dwconv.weight, reference.dwconv.bias, padding=3, groups=3
    ).permute(0, 2, 3, 1)
    rows = reference.act(reference.pwconv1(reference.norm(dense[active])))
    norm = torch.linalg.vector_norm(rows, dim=0, keepdim=True)
    ratio = norm / (norm.mean(-1, keepdim=True) + 1e-6)
    rows = (
        rows + reference.grn.gamma.reshape(1, -1) * rows * ratio + reference.grn.beta.reshape(1, -1)
    )
    rows = reference.pwconv2(rows) + (x_ref * valid).permute(0, 2, 3, 1)[active]
    torch.testing.assert_close(got.permute(0, 2, 3, 1)[active], rows)
    assert torch.equal(support, valid)
    assert torch.count_nonzero(got * (1 - valid)) == 0
    got.square().sum().backward()
    rows.square().sum().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    for a, b in zip(block.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


def test_sparse_downsample_layernorm_normalizes_channels_not_space():
    import torch
    import torch.nn.functional as f

    from owac.frameworks.n02_full.components.convnextv2 import LayerNorm
    from owac.frameworks.n02_full.components.fcmae import MaskAwareDenseConvNeXtV2Encoder

    x = torch.arange(36.0).reshape(1, 3, 3, 4)
    valid = torch.ones(1, 1, 3, 4)
    valid[:, :, 0, 0] = 0
    ln = LayerNorm(3, eps=1e-6, data_format="channels_first")
    got = MaskAwareDenseConvNeXtV2Encoder._masked_channels_first_layernorm(ln, x, valid)
    want = (
        f.layer_norm(x.permute(0, 2, 3, 1), (3,), ln.weight, ln.bias, 1e-6).permute(0, 3, 1, 2)
        * valid
    )
    torch.testing.assert_close(got, want)


def test_checkpoint_mean_std_uses_additive_epsilon_without_reordering(tmp_path):
    from owac.raw.rawvla_sequence_n02 import normalise_actions, normalise_state

    path = tmp_path / "stats.json"
    stats = {"mean": list(range(14)), "std": [0.0] + [0.4] * 13}
    path.write_text(json.dumps({"norm_stats": {"state": stats, "actions": stats}}))
    declared = d.load_formal_normalisation("robotwin", path)
    actions = np.ones((50, 14), dtype=np.float32)
    got, _, meta = normalise_actions(actions, None, declared, dataset_format="robotwin")
    expected = (actions - np.array(stats["mean"], dtype=np.float32)) / (
        np.array(stats["std"], dtype=np.float32) + np.float32(1e-8)
    )
    np.testing.assert_array_equal(got, expected)
    state, _ = normalise_state(actions[0], declared)
    np.testing.assert_array_equal(state, expected[0])
    assert meta["epsilon_mode"] == "add"
    assert len(meta["action_stats_sha256"]) == 64


def test_profile_without_required_modules_cannot_pass():
    evidence = {
        "inventory": {"status": "ok", "payload": {"status": "ok"}},
        "parity": {"status": "pass", "payload": {"status": "pass", "not_run_count": 0}},
        "profile": {"status": "ok", "payload": {"measured": {"targets": []}}},
    }
    assert p0.build_stage_decision(evidence, [])["implementation"] == "incomplete"

"""Regressions for defects found by the independent P0 acceptance audit."""

import json
from pathlib import Path

import pytest
import torch

from owac.raw import n01_full_manifest as manifest
from owac.runtime import n01_full_preflight as preflight
from owac.runtime.n01_full_profile import dbsr_factory

from .test_preflight import DEFAULT_COUNTS, _fixture_protocol, _write_json, _write_npz_tree

ROOT = Path(__file__).resolve().parents[3]


def test_strict_inventory_rejects_cross_split_source_identity(tmp_path):
    config = _fixture_protocol(tmp_path)
    p = tmp_path / "data/manifests/n01-full-v1/n01-full-v1-libero-manifest.json"
    data = json.loads(p.read_text())
    data["rows"][-1]["source_group"] = data["rows"][0]["source_group"]
    _write_json(p, data)
    protocol = json.loads(config.read_text())
    protocol["data"]["platform_manifests"]["libero"]["manifest_sha256"] = preflight.sha256_file(p)
    _write_json(config, protocol)
    result = preflight.inventory(config, tmp_path / "out", repo_root=tmp_path, data_root=None)
    assert "regions:libero:source_groups_disjoint" in result["summary"]["failures"]
    assert "manifest:libero:sha256" not in result["summary"]["failures"]


def test_strict_inventory_rejects_same_count_path_replacement(tmp_path):
    config = _fixture_protocol(tmp_path)
    root = tmp_path / "data-root"
    for platform, counts in DEFAULT_COUNTS.items():
        for kind in ["raw", "rgb"]:
            _write_npz_tree(root, platform, kind, sum(counts.values()))
    original = root / "libero/raw/train/libero-0.npz"
    original.rename(original.with_name("different-trajectory.npz"))
    result = preflight.inventory(config, tmp_path / "out", repo_root=tmp_path, data_root=root)
    failures = result["summary"]["failures"]
    assert "data_root:libero:exact_paths" in failures
    assert "data_root:libero:rgb_mirror_paths" in failures
    assert "data_root:libero:raw_npz_count" not in failures


def test_cost_uses_selected_resolution_cameras_and_accumulation(tmp_path):
    protocol = {
        "profiling": {
            "cost_model": {
                "basis_point_id": "formal",
                "units": [
                    {
                        "id": "matrix",
                        "updates": 1000,
                        "runs": 21,
                        "camera_count": 3,
                        "gradient_accumulation": 8,
                    }
                ],
            }
        }
    }
    for name, ms, resolution in [("tiny", 1, 64), ("formal", 50, 128)]:
        _write_json(
            tmp_path / "raw" / f"{name}.json",
            {
                "kind": "profile-point",
                "point_id": name,
                "resolution": resolution,
                "history": 8,
                "batch": 1,
                "channels": 4,
                "iterations": 20,
                "device_name": "fixture",
                "timings_ms": {"p50": ms, "p95": ms},
                "memory_bytes": {"peak_allocated": 1},
            },
        )
    result = preflight.profile_summarize(protocol, tmp_path / "raw", tmp_path / "out")
    assert result["gpu_hours_estimate"]["estimates"][0]["gpu_hours"] == pytest.approx(7.0)


def test_p0_factory_matches_existing_encoder_without_changing_old_history_guard():
    from owac.baselines.burst_features import BurstFeatureConfig, BurstFeatureEncoder

    torch.manual_seed(19)
    probe = dbsr_factory(protocol={}, point={"history": 8}, device="cpu")
    reference = BurstFeatureEncoder(BurstFeatureConfig())
    reference.load_state_dict(probe.encoder.state_dict())
    values = torch.rand(1, 8, 4, 4, 4)
    expected = reference(
        values.flip(1), torch.ones_like(values, dtype=torch.bool), torch.zeros(1, 8, 2, 4, 4)
    )[0]
    torch.testing.assert_close(probe(values), expected, rtol=0, atol=0)
    direct = dbsr_factory(protocol={}, point={"history": 16}, device="cpu")
    assert direct.entry == "direct"
    out = direct(torch.rand(1, 16, 4, 4, 4))
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in direct.parameters())
    assert sum(p.numel() for p in direct.parameters()) == 1497056
    with pytest.raises(ValueError, match="at most 8"):
        BurstFeatureEncoder(BurstFeatureConfig(max_history=16))
    with pytest.raises(ValueError, match="four-channel"):
        dbsr_factory(protocol={}, point={"history": 8, "channels": 3}, device="cpu")


def test_protocol_loader_adapter_binds_manifest_and_actual_statistics(tmp_path, monkeypatch):
    protocol = json.loads((ROOT / "configs/encoders/n01_full/protocol_v1.json").read_text())
    calls = []
    import owac.raw.rawvla_sequence as rawvla

    monkeypatch.setattr(rawvla, "RawVLASequenceDataset", lambda **kw: calls.append(kw))
    for platform in ["libero", "robotwin2"]:
        for region in manifest.REGIONS:
            manifest.dataset_from_protocol(protocol, platform, region, repo_root=ROOT, window=8)
            kw = calls[-1]
            assert kw["split"] == region
            assert kw["split_manifest"]["sides"][region]
    kw = calls[-1]
    assert kw["stats_key"] == ""
    assert kw["action_stats_key"] == "actions"
    # Actual server file shape: wrong protocol values used to raise a KeyError here.
    selected = rawvla.select_statistics(
        {"norm_stats": {"actions": {"mean": [0]}}},
        normalization="zscore",
        dataset_format="robotwin2",
        stats_key=kw["stats_key"],
    )
    assert kw["action_stats_key"] in selected

"""Offline teacher evaluator contract tests; no real model or rollout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from owac.evaluation.rawvla_teacher import deterministic_action_noise, ring_indices


def test_ring_indices_left_padding_is_causal_and_repeated_first():
    first = ring_indices(0)
    assert first.tolist() == [0, 0, 0, 0, 0, 0]
    later = ring_indices(12)
    assert later.tolist() == [7, 8, 9, 10, 11, 12]
    assert int(later.max()) == 12


def test_deterministic_noise_is_source_frame_bound():
    a = deterministic_action_noise(seed=101, source_id="s", frame=5)
    b = deterministic_action_noise(seed=101, source_id="s", frame=5)
    c = deterministic_action_noise(seed=101, source_id="s", frame=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (1, 10, 32)


def test_real_pinned_postprocessor_contract_when_source_present():
    source = Path(
        "outputs/raw-distillation-20261020-sol/starvla-pinned-source/examples/"
        "simBenchmarks/LIBERO/eval_files/openpi/pi_libero_common.py"
    )
    if not source.is_file():
        pytest.skip("pinned LIBERO postprocessor source is absent in this checkout")
    import sys

    root = source.parents[5]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("pinned_postprocess", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    norm = {
        "actions": {
            "q01": np.full(7, -1.0, dtype=np.float32),
            "q99": np.full(7, 1.0, dtype=np.float32),
        }
    }
    output = module.postprocess_openpi_actions(
        np.zeros((1, 32), dtype=np.float32),
        np.zeros((8,), dtype=np.float32),
        norm,
        model_name="PI05",
        model_source="openpi",
    )
    assert output.shape == (1, 7)
    assert np.isfinite(output).all()


def test_actual_validation_catalog_is_frozen_102_anchors():
    from pathlib import Path

    from owac.evaluation.rawvla_teacher import load_validation_catalog
    from owac.raw.teacher_data import load_source_manifest

    catalog = Path("outputs/rawvla-teacher-20261020/validation-catalog.json")
    manifest_path = Path("outputs/rawvla-teacher-20261020/raw/sources-audited.json")
    if not catalog.is_file() or not manifest_path.is_file():
        pytest.skip("actual validation catalog/source manifest absent")
    manifest = load_source_manifest(manifest_path, repo_root=Path("."), strict=True)
    anchors = load_validation_catalog(catalog, manifest)
    assert len(anchors) == 102
    assert len({source for source, _ in anchors}) == 5
    assert all(frame % 5 == 0 for _, frame in anchors)


def test_artifact_digest_tamper_is_rejected(tmp_path):
    from types import SimpleNamespace

    import torch

    from owac.evaluation.rawvla_teacher import (
        load_teacher_frontend_artifact,
        save_teacher_frontend_artifact,
    )

    class FakeFrontend(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            return {f"model.k{index}": torch.tensor([float(index)]) for index in range(83)}

    identity = {
        "backend": "PI05",
        "tokenizer": "tokenizer",
        "stats": "stats",
        "source": "source",
        "burst_frames": 6,
        "recurrent_endpoints": 6,
        "cameras": ["agentview", "wrist"],
        "cumulative_updates": 0,
        "input_provenance": "raw",
    }
    teacher = SimpleNamespace(frontend=FakeFrontend())
    path = tmp_path / "artifact.pt"
    save_teacher_frontend_artifact(teacher, path, identity=identity)
    loaded = load_teacher_frontend_artifact(path, expected_identity=identity)
    assert len(loaded) == 83
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["frontend_state_sha256"] = "0" * 64
    torch.save(payload, path)
    with pytest.raises(Exception, match="digest"):
        load_teacher_frontend_artifact(path, expected_identity=identity)

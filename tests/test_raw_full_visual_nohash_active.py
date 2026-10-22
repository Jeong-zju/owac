"""Active-path regression: no hashing may be invoked by the current pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from owac.baselines.raw_temporal_siglip import TemporalRawStemStudent
from owac.runtime.raw_full_visual_distillation import ImmutableTargetCache

_HELPER_PATH = Path(__file__).parent / "test_raw_temporal_siglip.py"
_SPEC = importlib.util.spec_from_file_location("nohash_temporal_helpers", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS)


def _forbid_hashes(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("active pipeline invoked a hash function")

    monkeypatch.setattr(hashlib, "sha256", forbidden)
    import owac.raw.teacher_data as teacher_data

    monkeypatch.setattr(teacher_data, "sha256_file", forbidden)
    import owac.baselines.raw_stem_siglip as raw_stem

    monkeypatch.setattr(raw_stem, "sha256_file", forbidden)
    monkeypatch.setattr(raw_stem, "sha256_tensor", forbidden)
    import owac.raw.temporal_distillation as temporal

    monkeypatch.setattr(temporal, "_payload_digest", forbidden)
    monkeypatch.setattr(temporal, "_sha256_text", forbidden)


def test_active_forward_backward_save_reload_without_hash(monkeypatch, tmp_path: Path) -> None:
    student = TemporalRawStemStudent(_HELPERS.fixture_student())
    reloaded = TemporalRawStemStudent(_HELPERS.fixture_student())
    packet = _HELPERS.make_packet(0, reset=True)
    _forbid_hashes(monkeypatch)
    output, state = student(packet, None)
    loss = output.projected_tokens.float().sum()
    loss.backward()
    assert state.model_identity.startswith("model-")
    assert "generation-" in state.model_identity

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    optimizer.step()
    scheduler.step()
    assert scheduler.state_dict()["last_epoch"] == 1
    torch.save(student.state_dict(), tmp_path / "model.pt")
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")
    reloaded.load_state_dict(torch.load(tmp_path / "model.pt", map_location="cpu"), strict=True)
    reload_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded.parameters() if parameter.requires_grad], lr=1e-3
    )
    reload_scheduler = torch.optim.lr_scheduler.StepLR(reload_optimizer, step_size=1)
    reload_optimizer.load_state_dict(
        torch.load(tmp_path / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    reload_scheduler.load_state_dict(
        torch.load(tmp_path / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    assert reload_scheduler.state_dict()["last_epoch"] == 1
    steps = [
        int(state["step"].item())
        for state in reload_optimizer.state_dict()["state"].values()
        if torch.is_tensor(state.get("step"))
    ]
    assert steps and all(step == 1 for step in steps)


def test_active_target_row_load_ignores_legacy_payload_digest(monkeypatch, tmp_path: Path) -> None:
    integration_path = Path(__file__).parent / "test_raw_temporal_integration.py"
    spec = importlib.util.spec_from_file_location("nohash_target_helpers", integration_path)
    helpers = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helpers)
    row = helpers.make_row()
    from owac.raw.temporal_distillation import CACHE_SCHEMA

    root = tmp_path / "cache"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "identity_key": "legacy-opaque-directory",
                "identity": {"legacy": "not-verified"},
            }
        )
    )
    directory = root / "legacy-opaque-directory" / row.source_id
    directory.mkdir(parents=True)
    torch.save(
        {
            "schema": CACHE_SCHEMA,
            "identity_key": "legacy-opaque-directory",
            "source_id": row.source_id,
            "endpoint": row.endpoint,
            "row": row.to_payload(),
        },
        directory / f"{row.endpoint:06d}.pt",
    )
    cache = ImmutableTargetCache(root, expected_identity_key="ignored")
    _forbid_hashes(monkeypatch)
    loaded = cache.read(row.source_id, row.endpoint)
    torch.testing.assert_close(loaded.teacher_tokens, row.teacher_tokens)
    assert loaded.prefix_sha256 == row.prefix_sha256


def test_nohash_config_has_no_digest_assets_and_builds_asset_set(monkeypatch) -> None:
    import json

    from owac.raw.temporal_distillation import TemporalAssetSet
    from owac.runtime.raw_full_visual_distillation import _data_contract

    config = json.loads(
        Path(
            "outputs/raw-distillation-full-vision-20261022/shared-stage/config-nohash.json"
        ).read_text()
    )
    for key in (
        "artifact_sha256",
        "backend_sha256",
        "tokenizer_sha256",
        "stats_sha256",
        "sources_sha256",
        "source_manifest_sha256",
        "checkpoint_sha256",
        "statistics_sha256",
        "teacher_frontend_checkpoint_sha256",
    ):
        assert key not in (config.get("assets") or {})
    assert _data_contract(config)["expected_splits"] == {"train": 8, "validation": 2, "test": 2}
    _forbid_hashes(monkeypatch)
    assets = TemporalAssetSet.from_config(config, repo_root=Path.cwd())
    assert assets.sources_path.name == "source-manifest.json"

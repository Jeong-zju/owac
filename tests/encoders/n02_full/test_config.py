"""Config contract tests for the N02 full (v1.1) line.

They pin the frozen field defaults, the JSON loading rules (missing keys keep
defaults; unknown keys and type mismatches name the offending key), the
canonical hash's stability under dict ordering, and the semantic checks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from owac.frameworks.n02_full import config as cfg_mod

REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs/encoders/n02_full/p0.json"


def _write_config(tmp_path: Path, payload: object) -> Path:
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_defaults_match_the_frozen_contract() -> None:
    cfg = cfg_mod.N02FullConfig()
    assert cfg_mod.CONFIG_VERSION == "n02-full-p0-v1"
    assert cfg.version == "n02-full-v1.1"
    assert cfg.seeds == (1103, 2207, 3301)
    assert (cfg.worlds, cfg.slots, cfg.slot_dim) == (8, 16, 512)
    assert cfg.depths == (3, 3, 27, 3)
    assert cfg.dims == (128, 256, 512, 1024)
    assert cfg.input_size == 256
    assert cfg.scales == ("stride8", "stride16")
    assert cfg.partitions == {"train": 0.8, "dev": 0.1, "calibration": 0.1}
    assert cfg.data == {
        "cache_root": None,
        "dataset_format": None,
        "protocol_verified": False,
        "notes": "",
    }
    assert {k: cfg.runtime[k] for k in ("dtype", "device", "amp")} == {
        "dtype": "float32",
        "device": "cpu",
        "amp": False,
    }
    assert cfg.runtime["acceptance_budgets"]["decision_p95_ms_per_episode"] == 200
    assert cfg.runtime["acceptance_budgets"]["persistent_state_bytes_per_episode"] == 256 * 2**20
    assert cfg.profile == {
        "updates": 200,
        "warmup_updates": 20,
        "batch_size": 2,
        "sequence_length": 8,
        "operation": "train_step",
    }
    assert cfg.paths == {"output_root": "outputs/n02-full"}


def test_shipped_config_equals_defaults_and_validates() -> None:
    cfg = cfg_mod.load_config(REPO_CONFIG)
    assert cfg == cfg_mod.N02FullConfig()
    assert cfg_mod.validate_config(cfg) == []
    assert cfg_mod.config_hash(cfg) == cfg_mod.config_hash(cfg_mod.N02FullConfig())


def test_missing_keys_keep_defaults(tmp_path: Path) -> None:
    cfg = cfg_mod.load_config(_write_config(tmp_path, {"worlds": 4, "scales": ["stride8"]}))
    assert cfg.worlds == 4
    assert cfg.scales == ("stride8",)
    assert cfg.seeds == cfg_mod.N02FullConfig().seeds
    assert cfg.partitions == cfg_mod.N02FullConfig().partitions


def test_tuple_fields_accept_json_lists(tmp_path: Path) -> None:
    cfg = cfg_mod.load_config(
        _write_config(tmp_path, {"seeds": [7, 11], "depths": [1, 1, 1], "dims": [2, 3]})
    )
    assert cfg.seeds == (7, 11)
    assert cfg.depths == (1, 1, 1)
    assert cfg.dims == (2, 3)
    assert cfg_mod.validate_config(cfg)  # depths/dims lengths differ: reported, not raised


def test_unknown_key_is_rejected_and_named(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"world": 4})
    with pytest.raises(cfg_mod.ConfigError) as excinfo:
        cfg_mod.load_config(path)
    assert "world" in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert isinstance(excinfo.value, TypeError)


@pytest.mark.parametrize(
    ("payload", "key"),
    [
        ({"worlds": "8"}, "worlds"),
        ({"worlds": True}, "worlds"),
        ({"seeds": "1103"}, "seeds"),
        ({"seeds": [1103, "2207"]}, "seeds"),
        ({"depths": [3, 3.5]}, "depths"),
        ({"scales": [1, 2]}, "scales"),
        ({"scales": "stride8"}, "scales"),
        ({"partitions": [0.5, 0.5]}, "partitions"),
        ({"partitions": {"train": "0.8"}}, "partitions.train"),
        ({"runtime": {"dtype": 32}}, "runtime.dtype"),
        ({"runtime": {"amp": "no"}}, "runtime.amp"),
        ({"profile": {"updates": 1.5}}, "profile.updates"),
        ({"paths": {"output_root": 3}}, "paths.output_root"),
        ({"data": {"cache_root": 7}}, "data.cache_root"),
        ({"data": {"protocol_verified": "yes"}}, "data.protocol_verified"),
        ({"version": 1}, "version"),
    ],
)
def test_type_mismatch_is_rejected_and_names_the_key(
    tmp_path: Path, payload: dict, key: str
) -> None:
    path = _write_config(tmp_path, payload)
    with pytest.raises(cfg_mod.ConfigError) as excinfo:
        cfg_mod.load_config(path)
    assert key in str(excinfo.value)


def test_non_object_and_missing_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load_config(_write_config(tmp_path, [1, 2, 3]))
    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load_config(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load_config(broken)


def test_as_dict_is_json_ready_and_does_not_alias_the_config() -> None:
    cfg = cfg_mod.N02FullConfig()
    payload = cfg_mod.as_dict(cfg)
    assert isinstance(payload["seeds"], list)
    assert isinstance(payload["depths"], list)
    payload["partitions"]["train"] = 0.0
    payload["data"]["notes"] = "mutated"
    assert cfg.partitions["train"] == 0.8
    assert cfg.data["notes"] == ""
    json.dumps(payload)


def test_config_hash_is_stable_and_order_independent(tmp_path: Path) -> None:
    first = cfg_mod.load_config(
        _write_config(tmp_path, {"partitions": {"train": 0.8, "dev": 0.1, "calibration": 0.1}})
    )
    second = cfg_mod.load_config(
        _write_config(tmp_path, {"partitions": {"calibration": 0.1, "train": 0.8, "dev": 0.1}})
    )
    assert cfg_mod.config_hash(first) == cfg_mod.config_hash(second)
    changed = cfg_mod.load_config(_write_config(tmp_path, {"worlds": 9}))
    assert cfg_mod.config_hash(changed) != cfg_mod.config_hash(first)
    assert len(cfg_mod.config_hash(first)) == 64


def test_validate_config_accepts_defaults_and_reports_specific_issues() -> None:
    assert cfg_mod.validate_config(cfg_mod.N02FullConfig()) == []

    cases = {
        "partitions: weights must sum to 1": cfg_mod.N02FullConfig(
            partitions={"train": 0.5, "dev": 0.1, "calibration": 0.1}
        ),
        "partitions: missing partition(s) calibration": cfg_mod.N02FullConfig(
            partitions={"train": 0.5, "dev": 0.5}
        ),
        "depths/dims: must have equal length": cfg_mod.N02FullConfig(depths=(3, 3, 27)),
        "worlds: must be positive": cfg_mod.N02FullConfig(worlds=0),
        "slots: must be positive": cfg_mod.N02FullConfig(slots=-1),
        "slot_dim: must be positive": cfg_mod.N02FullConfig(slot_dim=0),
        "input_size: must be positive": cfg_mod.N02FullConfig(input_size=0),
        "scales: must contain 'stride16'": cfg_mod.N02FullConfig(scales=("stride8",)),
        "seeds: must be non-empty": cfg_mod.N02FullConfig(seeds=()),
        "runtime.dtype: must be one of": cfg_mod.N02FullConfig(
            runtime={"dtype": "float64", "device": "cpu", "amp": False}
        ),
    }
    for expected, cfg in cases.items():
        issues = cfg_mod.validate_config(cfg)
        assert issues, expected
        assert any(expected in issue for issue in issues), (expected, issues)

    # Directly constructed garbage is reported, never raised.
    assert cfg_mod.validate_config(cfg_mod.N02FullConfig(partitions=[]))
    assert cfg_mod.validate_config(cfg_mod.N02FullConfig(depths=("a",), dims=(1,)))
    assert cfg_mod.validate_config(cfg_mod.N02FullConfig(runtime=None))


def test_package_import_stays_light_and_does_not_eager_import_submodules() -> None:
    script = (
        "import sys; import owac.frameworks.n02_full as pkg; "
        "assert pkg.__all__ == "
        "['config', 'state', 'data', 'synthetic', 'components'], pkg.__all__; "
        "assert 'owac.frameworks.n02_full.config' not in sys.modules, 'config eagerly imported'; "
        "assert 'owac.frameworks.n02_full.state' not in sys.modules, 'state eagerly imported'; "
        "assert 'owac.frameworks.n02_full.data' not in sys.modules, 'data eagerly imported'; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'numpy' not in sys.modules, 'numpy imported'; "
        "from owac.frameworks.n02_full import config; "
        "print(config.CONFIG_VERSION)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "n02-full-p0-v1"

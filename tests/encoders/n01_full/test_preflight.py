"""Behavioural tests for the N01 v1.0 P0 preflight entry point.

These tests use independent oracles (standard SHA256 vectors, hand-computed
aggregates, tampered bundles, B's frozen manifests) instead of re-running the
implementation's own assumptions, so a broken version actually fails.  The
upstream-lock oracle works both with and without the ignored local upstream
cache, and it fails when the recorded ``hash_source`` is not honest about what
could actually be recomputed.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from owac.runtime import n01_full_preflight as preflight

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = REPO_ROOT / "configs/encoders/n01_full/protocol_v1.json"
LOCK_PATH = REPO_ROOT / "configs/encoders/n01_full/upstream_lock.json"
MANIFEST_DIR = REPO_ROOT / "data/manifests/n01-full-v1"
UPSTREAM_CACHE = REPO_ROOT / "outputs/n01-v1-prompt-preparation/upstream"
BACKEND_DIR = REPO_ROOT / "configs/backends/n01-full-runtime"

# Independent oracle: standard SHA256 test vectors, not computed through the module.
SHA256_VECTORS = {
    b"": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    b"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
}

DEFAULT_COUNTS = {
    "libero": {"train": 4, "calibration": 2, "validation": 2, "test": 2},
    "robotwin2": {"train": 2, "calibration": 1, "validation": 1, "test": 1},
}


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_platform_manifests(root: Path, counts: dict) -> dict[str, Path]:
    paths = {}
    for platform, region_counts in counts.items():
        sides = {
            region: [f"{region}/{platform}-{index}.npz" for index in range(size)]
            for region, size in region_counts.items()
        }
        rows = [
            {
                "relative_path": path,
                "split": region,
                "dataset_id": platform,
                "input_mode": (
                    "raw_uint8_pseudo_raw" if platform == "libero" else "raw_float32_hdr_derived"
                ),
                "source_group": f"{platform}-{region}-{index}",
                "episode_id": f"{platform}-episode-{index}",
                "camera_id": "front",
                "units": "pseudo_raw_hdr",
                "calibration_version": f"{platform}-cal-v1",
            }
            for region, region_paths in sides.items()
            for index, path in enumerate(region_paths)
        ]
        payload = {
            "schema": "owac.n01_full.manifest.v1",
            "sides": sides,
            "rows": rows,
            "regions": list(region_counts),
        }
        paths[platform] = _write_json(
            root / f"data/manifests/n01-full-v1/n01-full-v1-{platform}-manifest.json", payload
        )
    return paths


def _fixture_protocol(
    root: Path,
    counts: dict = DEFAULT_COUNTS,
    *,
    declared_hash: str | None = None,
    expected: dict | None = None,
) -> Path:
    manifest_paths = _write_platform_manifests(root, counts)
    splits = {}
    for region in preflight.REQUIRED_SPLITS:
        splits[region] = {
            "platform_manifests": {
                platform: {
                    "path": str(manifest_paths[platform].relative_to(root)),
                    "sha256": declared_hash or preflight.sha256_file(manifest_paths[platform]),
                }
                for platform in counts
            },
            "counts": {platform: counts[platform][region] for platform in counts},
            "cross_platform_total": sum(counts[platform][region] for platform in counts),
        }
    protocol = {
        "protocol_version": "fixture",
        "stage": "P0",
        "data": {
            "data_root": str(root / "data-root"),
            "raw_roots": {platform: f"{platform}/raw" for platform in counts},
            "rgb_mirror_roots": {platform: f"{platform}/rgb" for platform in counts},
            "platforms": {
                platform: {
                    "expected_npz_frozen": (expected or {}).get(
                        platform, sum(counts[platform].values())
                    )
                }
                for platform in counts
            },
            "platform_manifests": {
                platform: {
                    "manifest_path": str(manifest_paths[platform].relative_to(root)),
                    "manifest_sha256": (
                        declared_hash or preflight.sha256_file(manifest_paths[platform])
                    ),
                }
                for platform in counts
            },
            "splits": splits,
        },
    }
    return _write_json(root / "configs/encoders/n01_full/protocol_v1.json", protocol)


def _write_npz_tree(root: Path, platform: str, kind: str, size: int) -> None:
    directory = root / f"{platform}/{kind}"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(size):
        (directory / f"{kind}-{index}.npz").write_bytes(b"npz")


def _build_mini_source(root: Path) -> None:
    (root / "src/owac/pkg").mkdir(parents=True, exist_ok=True)
    (root / "src/owac/pkg/__init__.py").write_text("", encoding="utf-8")
    (root / "src/owac/pkg/tracked_module.py").write_text("TRACKED = 1\n", encoding="utf-8")
    (root / "src/owac/runtime").mkdir(parents=True, exist_ok=True)
    (root / "src/owac/runtime/n01_full_preflight.py").write_text(
        "# closure stub\n", encoding="utf-8"
    )
    _write_json(
        root / "configs/encoders/n01_full/protocol_v1.json",
        {key: {} for key in preflight.REQUIRED_PROTOCOL_KEYS} | {"protocol_version": "fixture"},
    )
    _write_json(root / "configs/encoders/n01_full/upstream_lock.json", {"repositories": []})
    _write_json(
        root / "data/manifests/n01-full-v1/n01-full-v1-index.json",
        {"schema": "owac.n01_full.manifest_index.v1"},
    )
    _write_json(
        root / "data/manifests/n01-full-v1/n01-full-v1-libero-manifest.json",
        {"schema": "owac.n01_full.manifest.v1", "sides": {}},
    )
    _write_json(
        root / "data/manifests/n01-full-v1/n01-full-v1-robotwin2-manifest.json",
        {"schema": "owac.n01_full.manifest.v1", "sides": {}},
    )
    (root / "configs/backends/n01-full-runtime").mkdir(parents=True, exist_ok=True)
    (root / "configs/backends/n01-full-runtime/requirements-gpu.txt").write_text(
        "torch==2.7.1+cu128\n", encoding="utf-8"
    )
    (root / "tests/encoders/n01_full").mkdir(parents=True, exist_ok=True)
    (root / "tests/encoders/n01_full/test_fixture.py").write_text("", encoding="utf-8")
    (root / "docs/prompts/n01_v1_0").mkdir(parents=True, exist_ok=True)
    _write_json(root / "docs/prompts/n01_v1_0/source_manifest.json", {"repositories": []})
    # denied paths that must never enter the bundle
    (root / ".venv/lib").mkdir(parents=True, exist_ok=True)
    (root / ".venv/lib/site.py").write_text("x = 1\n", encoding="utf-8")
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "outputs/run.json").write_text("{}\n", encoding="utf-8")
    (root / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (root / "secrets.json").write_text("{}\n", encoding="utf-8")
    (root / "src/owac/pkg/model.safetensors").write_text("weights\n", encoding="utf-8")
    (root / "src/owac/pkg/__pycache__").mkdir(parents=True, exist_ok=True)
    (root / "src/owac/pkg/__pycache__/x.pyc").write_text("bytecode\n", encoding="utf-8")


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _rewrite_archive(archive: Path, *, drop: str | None = None, extra: str | None = None) -> None:
    with tarfile.open(archive, "r:gz") as source:
        members = [
            (member.name, source.extractfile(member).read())
            for member in source.getmembers()
            if member.isfile()
        ]
    temporary = archive.with_suffix(".tmp.gz")
    with tarfile.open(temporary, "w:gz") as target:
        for name, data in members:
            if name == drop:
                continue
            info = tarfile.TarInfo(name)
            info.size = len(data)
            target.addfile(info, io.BytesIO(data))
        if extra is not None:
            payload = b"injected"
            info = tarfile.TarInfo(extra)
            info.size = len(payload)
            target.addfile(info, io.BytesIO(payload))
    temporary.replace(archive)


# ---------------------------------------------------------------------------------------
# hashing and CLI surface
# ---------------------------------------------------------------------------------------


def test_sha256_file_matches_standard_vectors(tmp_path: Path) -> None:
    for payload, expected in SHA256_VECTORS.items():
        target = tmp_path / "blob.bin"
        target.write_bytes(payload)
        assert preflight.sha256_file(target) == expected, payload[:16]


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["inventory", "--help"],
        ["profile", "--help"],
        ["bundle", "--help"],
        ["verify", "--help"],
    ],
)
def test_help_for_every_subcommand(argv: list[str], capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        preflight.main(argv)
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_module_entry_point_help_runs_as_subprocess() -> None:
    environment = {"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"}
    completed = subprocess.run(
        [sys.executable, "-m", "owac.runtime.n01_full_preflight", "verify", "--help"],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--bundle" in completed.stdout


# ---------------------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------------------


def test_inventory_four_regions_from_platform_manifests(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _fixture_protocol(root)
    result = preflight.inventory(config, tmp_path / "out", repo_root=root, data_root=None)

    assert result["summary"]["failures"] == []
    assert result["summary"]["missing_manifests"] == []
    assert result["summary"]["region_totals"] == {
        "train": 6,
        "calibration": 3,
        "validation": 3,
        "test": 3,
    }
    assert result["summary"]["cross_platform_total"] == 15
    assert result["platform_manifests"]["libero"]["region_counts"] == {
        "train": 4,
        "calibration": 2,
        "validation": 2,
        "test": 2,
    }
    assert result["regions"]["train"]["datasets"] == {"libero": 4, "robotwin2": 2}
    assert result["regions"]["train"]["storage_encodings"] == {
        "raw_uint8_pseudo_raw": 4,
        "raw_float32_hdr_derived": 2,
    }
    assert result["regions"]["train"]["camera_count"] == 1
    assert result["regions"]["train"]["cross_platform_total"] == 6
    assert result["regions"]["train"]["declared_cross_platform_total"] == 6
    # the local data root does not exist here, so its checks stay pending, not silently ok
    assert "data_root:present" in result["summary"]["pending"]
    assert (tmp_path / "out/inventory.json").is_file()


def test_inventory_counts_raw_roots_only_and_reports_rgb_mirror(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _fixture_protocol(root)
    data_root = root / "data-root"
    for platform, counts in DEFAULT_COUNTS.items():
        total = sum(counts.values())
        _write_npz_tree(data_root, platform, "raw", total)
        _write_npz_tree(data_root, platform, "rgb", total)

    result = preflight.inventory(config, tmp_path / "out", repo_root=root, data_root=data_root)
    assert result["summary"]["failures"] == []
    assert result["data_root"]["raw_counts"]["libero"]["count"] == 10
    assert result["data_root"]["rgb_mirror_counts"]["libero"]["count"] == 10
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["data_root:libero:raw_npz_count"]["ok"] is True
    assert by_id["data_root:libero:rgb_mirror_1to1"]["ok"] is True

    # a 2x counting bug would make the raw count 16 here; drop one mirror file to prove
    # the mirror is checked separately from the raw measurement count
    (data_root / "libero/rgb/rgb-0.npz").unlink()
    result = preflight.inventory(config, tmp_path / "out2", repo_root=root, data_root=data_root)
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["data_root:libero:raw_npz_count"]["ok"] is True
    assert by_id["data_root:libero:raw_npz_count"]["detail"].startswith("observed=10")
    assert by_id["data_root:libero:rgb_mirror_1to1"]["ok"] is False
    assert result["summary"]["status"] == "failed"


def test_inventory_flags_overlap_union_and_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _fixture_protocol(root)
    manifest_path = root / "data/manifests/n01-full-v1/n01-full-v1-libero-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["sides"]["validation"] = [*payload["sides"]["validation"], payload["sides"]["train"][0]]
    payload["sides"]["test"] = payload["sides"]["test"][:1]
    payload["rows"] = [
        row
        for row in payload["rows"]
        if row["relative_path"] in {path for paths in payload["sides"].values() for path in paths}
    ]
    _write_json(manifest_path, payload)

    result = preflight.inventory(config, tmp_path / "out", repo_root=root, data_root=None)
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["regions:libero:disjoint"]["ok"] is False
    assert "train|validation" in by_id["regions:libero:disjoint"]["detail"]
    assert by_id["regions:libero:union"]["ok"] is False
    # the protocol still carries the old hash, so the manifest hash must mismatch
    assert by_id["manifest:libero:sha256"]["ok"] is False
    assert result["summary"]["status"] == "failed"

    # a declared per-platform total that no longer matches also fails (no silent refresh)
    other = tmp_path / "other"
    config2 = _fixture_protocol(other, expected={"libero": 99, "robotwin2": 6})
    result2 = preflight.inventory(config2, tmp_path / "out2", repo_root=other, data_root=None)
    by_id2 = {check["id"]: check for check in result2["checks"]}
    assert by_id2["regions:libero:union"]["ok"] is False
    assert "expected=99" in by_id2["regions:libero:union"]["detail"]


def test_inventory_missing_manifest_checks_stay_pending(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _fixture_protocol(root)
    (root / "data/manifests/n01-full-v1/n01-full-v1-robotwin2-manifest.json").unlink()
    result = preflight.inventory(config, tmp_path / "out", repo_root=root, data_root=None)
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["manifest:robotwin2:present"]["ok"] is False
    for check_id in (
        "region:robotwin2:train:nonempty",
        "regions:robotwin2:disjoint",
        "regions:robotwin2:union",
    ):
        assert by_id[check_id]["ok"] is None, check_id
    assert by_id["region:libero:train:nonempty"]["ok"] is True
    assert result["summary"]["status"] == "failed"


def test_inventory_requires_platform_manifest_declaration(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _write_json(
        root / "configs/encoders/n01_full/protocol_v1.json",
        {"protocol_version": "fixture", "stage": "P0", "data": {}},
    )
    result = preflight.inventory(config, tmp_path / "out", repo_root=root, data_root=None)
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["protocol:platform_manifests_declared"]["ok"] is False
    assert result["summary"]["status"] == "failed"


def test_inventory_strict_cli_exit_codes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    config = _fixture_protocol(root, declared_hash="0" * 64)
    common = ["--config", str(config), "--repo-root", str(root)]
    assert preflight.main(["inventory", *common, "--output", str(tmp_path / "a")]) == 0
    assert (
        preflight.main(["inventory", *common, "--output", str(tmp_path / "b"), "--strict"])
        == preflight.EXIT_FAILED
    )


def test_inventory_against_the_frozen_platform_manifests(tmp_path: Path) -> None:
    """The protocol must match B's frozen manifests: hashes, counts, disjointness."""
    if not MANIFEST_DIR.is_dir():
        pytest.skip("frozen platform manifests are not present in this checkout")
    result = preflight.inventory(
        PROTOCOL_PATH, tmp_path / "out", repo_root=REPO_ROOT, data_root=None
    )
    assert result["summary"]["failures"] == [], result["summary"]["failures"]
    assert result["platform_manifests"]["libero"]["region_counts"] == {
        "train": 1228,
        "calibration": 175,
        "validation": 206,
        "test": 162,
    }
    assert result["platform_manifests"]["robotwin2"]["region_counts"] == {
        "train": 450,
        "calibration": 48,
        "validation": 58,
        "test": 76,
    }
    assert result["summary"]["region_totals"] == {
        "train": 1678,
        "calibration": 223,
        "validation": 264,
        "test": 238,
    }
    assert result["summary"]["cross_platform_total"] == 2403
    by_id = {check["id"]: check for check in result["checks"]}
    assert by_id["regions:libero:union"]["ok"] is True
    assert by_id["regions:robotwin2:union"]["ok"] is True
    assert by_id["manifest:libero:sha256"]["ok"] is True
    assert by_id["manifest:robotwin2:sha256"]["ok"] is True
    # the data root is remote; the local run must leave those checks pending
    assert "data_root:present" in result["summary"]["pending"]


# ---------------------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------------------


def test_profile_collect_refuses_pending_p4_factory(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "profile"
    with pytest.raises(preflight.PreflightUnavailable) as excinfo:
        preflight.profile_collect(protocol, output, raw_dir=output / "raw")
    message = str(excinfo.value)
    assert "pending_P4" in message and "P4" in message
    assert not (output / "profile-summary.json").exists()
    exit_code = preflight.main(
        [
            "profile",
            "--config",
            str(PROTOCOL_PATH),
            "--output",
            str(tmp_path / "profile-cli"),
            "--mode",
            "collect",
        ]
    )
    assert exit_code == preflight.EXIT_UNAVAILABLE


def test_profile_collect_refuses_cpu_with_explicit_factory(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "profile"
    with pytest.raises((preflight.PreflightRefused, preflight.PreflightUnavailable)) as excinfo:
        preflight.profile_collect(
            protocol,
            output,
            raw_dir=output / "raw",
            factory_spec="owac.runtime.n01_full_preflight:sha256_file",
        )
    message = str(excinfo.value)
    assert "CUDA" in message or "torch" in message
    assert not (output / "profile-summary.json").exists()
    raw_dir = output / "raw"
    if raw_dir.is_dir():
        assert not list(raw_dir.glob("*.json"))


def test_profile_backbone_contract_is_enforced() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    resolved = preflight.validate_profiling_backbone(protocol)
    assert resolved["backbone"] == "dbsr"
    assert resolved["contract"]["channels"] == 64
    assert resolved["contract"]["feature_width"] == 512

    shrunk = json.loads(json.dumps(protocol))
    shrunk["profiling"]["backbone_contract"]["channels"] = 32
    with pytest.raises(preflight.PreflightRefused, match="non-full-size DBSR"):
        preflight.validate_profiling_backbone(shrunk)
    # An explicit backbone override is the sanctioned route, not relaxing the sizes.
    assert preflight.validate_profiling_backbone(shrunk, "bsrt")["backbone"] == "bsrt"
    with pytest.raises(preflight.PreflightRefused, match="unknown profiling backbone"):
        preflight.validate_profiling_backbone(protocol, "resnet")


def test_profile_summarize_aggregates_known_points(tmp_path: Path) -> None:
    protocol = {
        "profiling": {
            "measurement_channels": 4,
            "cost_model": {
                "units": [
                    {"id": "with_runs", "updates": 1000, "runs": 2},
                    {"id": "pending_runs", "updates": 1000, "runs": "pending_profiling"},
                ]
            },
        }
    }
    raw = tmp_path / "raw"
    _write_json(
        raw / "p-1.json",
        {
            "kind": "profile-point",
            "point_id": "p-1",
            "backbone": "dbsr",
            "resolution": 128,
            "history": 8,
            "batch": 1,
            "channels": 4,
            "iterations": 20,
            "device_name": "fixture",
            "timings_ms": {"p50": 10.0, "p95": 20.0, "min": 9.0, "max": 21.0, "mean": 12.0},
            "memory_bytes": {"peak_allocated": 1024},
        },
    )
    _write_json(
        raw / "p-2.json",
        {
            "kind": "profile-point",
            "point_id": "p-1",
            "backbone": "dbsr",
            "resolution": 128,
            "history": 8,
            "batch": 1,
            "channels": 4,
            "iterations": 20,
            "device_name": "fixture",
            "timings_ms": {"p50": 20.0, "p95": 30.0, "min": 19.0, "max": 31.0, "mean": 22.0},
            "memory_bytes": {"peak_allocated": 2048},
        },
    )
    result = preflight.profile_summarize(protocol, raw, tmp_path / "out")
    assert len(result["points"]) == 1
    point = result["points"][0]
    assert point["samples"] == 2
    assert point["timings_ms"]["p50"] == pytest.approx(15.0)
    assert point["memory_bytes"]["peak_allocated"] == 2048
    estimates = {item["unit_id"]: item for item in result["gpu_hours_estimate"]["estimates"]}
    # step_seconds = 0.015 s; 1000 updates x 2 runs -> 30 s -> 1/120 GPU hours
    assert estimates["with_runs"]["gpu_hours"] == pytest.approx(30.0 / 3600.0)
    assert estimates["pending_runs"]["gpu_hours"] is None
    assert "pending" in estimates["pending_runs"]["note"]
    assert (tmp_path / "out/profile-summary.json").is_file()

    with pytest.raises(preflight.PreflightUnavailable):
        preflight.profile_summarize(protocol, tmp_path / "empty", tmp_path / "out2")


# ---------------------------------------------------------------------------------------
# protocol and upstream lock contracts
# ---------------------------------------------------------------------------------------


def test_protocol_carries_required_contract_fields() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    missing = [key for key in preflight.REQUIRED_PROTOCOL_KEYS if key not in protocol]
    assert missing == []
    identity = protocol["input_identity"]
    assert identity["time_order"] == "oldest_first"
    assert identity["fusion_time_order"] == "newest_first"
    assert identity["time_order_adapter"] == "oldest_first_to_newest_first"
    assert set(identity["input_modes"]) == {"bayer_cfa", "synthetic_cfa", "rawvla_rgb3"}
    naming = identity["identity_naming_rule"]
    assert "unknown_metadata" in naming["metadata_free"]
    assert "non-empty" in naming["rule"]
    assert "unknown_metadata" in identity["camera_id_vocabulary"]["values"]

    assert protocol["coordinate_units"]["coordinate_convention"] == "pull"
    assert protocol["coordinate_units"]["input_channels_multiple_rule"]

    replay = protocol["exact_replay"]
    assert replay["exact_window_unique_frames"] == 32
    assert replay["unique_frames_choices"] == [8, 16, 32]

    content = protocol["models_and_upstream"]["content_backbone"]
    assert (content["channels"], content["num_blocks"], content["feature_width"]) == (64, 9, 512)
    assert content["readout_tokens"] == 256 and content["readout_width"] == 2048
    assert content["time_order_adapter"] == "oldest_first_to_newest_first"
    readout = protocol["models_and_upstream"]["readout"]
    assert readout["bridge_in_dim_rule"].endswith("(=512)")
    assert readout["bridge_width_rule"].endswith("(=2048)")

    seeds = protocol["seeds"]
    assert seeds["model"]["values"] == [101, 211, 307]
    assert seeds["data"]["values"], "data seeds must be listed separately"
    assert not set(seeds["model"]["values"]) & set(seeds["data"]["values"])
    assert seeds["separated"] is True

    data = protocol["data"]
    assert data["raw_roots"] and data["rgb_mirror_roots"]
    assert data["count_rule"]
    assert set(data["platform_manifests"]) == {"libero", "robotwin2"}
    assert "input_mode" not in data["platforms"]["libero"]
    assert data["platforms"]["libero"]["model_input_mode"] == "rawvla_rgb3"
    layers = data["model_input_mode_vs_storage_encoding"]
    assert layers["storage_encoding"]["location"]
    assert "input_mode" in layers["storage_encoding"]["location"]
    for region in preflight.REQUIRED_SPLITS:
        entry = data["splits"][region]
        assert entry["counts"] and entry["cross_platform_total"]
        assert set(entry["platform_manifests"]) == {"libero", "robotwin2"}

    assert protocol["profiling"]["encoder_factory"] == "pending_P4"
    assert protocol["stage_status"]["state"] == "preflight_backfilled_pending_remote_strict_run"
    assert protocol["stage_status"]["state_note"]
    serialized = json.dumps(protocol)
    assert "pending_manifest_hash_" not in serialized
    assert "pending_weights_hash" not in serialized
    assert "pending_stats_hash" not in serialized
    assert "pending_tokenizer_hash" not in serialized
    assert "pending_calibration" in serialized
    assert protocol["stage_status"]["pending_fields"] == [
        "metrics.planned_gates.*.value",
        "profiling.encoder_factory",
        "statistics.alpha",
        "statistics.min_paired_units",
        "thresholds.fields.*",
    ]

    # the backfilled values are present with an explicit provenance status
    profiling_budget = protocol["budget"]["pending_profiling"]
    assert profiling_budget["status"] == "filled_from_p0_measurement"
    assert profiling_budget["resolution"]["value"] == 128
    assert profiling_budget["step_time_seconds"]["value"] == pytest.approx(0.046073)
    assert profiling_budget["step_time_seconds"]["value_status"] == "measured"
    assert profiling_budget["measured_micro_batch"]["value"] == 1
    assert profiling_budget["effective_batch"]["value"] == 8
    assert (
        profiling_budget["effective_batch"]["value_status"] == "planned_from_measured_micro_batch"
    )
    assert profiling_budget["max_steps"]["value"] == 100000
    assert profiling_budget["io_mib_per_second"]["throughput"] == "derived_pending"
    for asset in (
        protocol["data"]["stats"]["libero"],
        protocol["data"]["stats"]["robotwin2"],
        protocol["backend_protocol"]["tokenizer"],
        protocol["backend_protocol"]["weights"]["libero"],
        protocol["backend_protocol"]["weights"]["robotwin2"],
    ):
        assert len(asset["sha256"]) == 64
        assert asset["value_status"] == "measured_by_lead_2026-10-14"
        assert asset["source"]
    assert protocol["seeds"]["data"]["value_status"].startswith("confirmed_by_lead")


def test_protocol_gpu_hours_estimate_is_internally_consistent() -> None:
    """Recompute the priced GPU hours from the recorded measured step time."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    budget = protocol["budget"]["pending_profiling"]
    step = budget["step_time_seconds"]["value"]
    hours = budget["gpu_hours"]
    runs = hours["runs"]
    per_update = step * (runs["libero_runs"] * 2 + runs["robotwin2_runs"] * 3)
    for updates, key in ((20000, "twenty_k"), (100000, "hundred_k"), (300000, "three_hundred_k")):
        assert hours[key] == pytest.approx(per_update * updates / 3600.0, rel=1e-3), key
    assert hours["value_status"] == "estimated_from_measured_step_time"
    variants = hours["variants"]
    six = variants["six_run_minimum_start"]
    per_update_six = step * (3 * 2 + 3 * 3)
    for updates, key in ((20000, "twenty_k"), (100000, "hundred_k"), (300000, "three_hundred_k")):
        assert six[key] == pytest.approx(per_update_six * updates / 3600.0, rel=1e-3), key
    batch_eight = variants["effective_batch_8"]
    assert batch_eight["hundred_k"] == pytest.approx(hours["hundred_k"] * 8, rel=1e-3)
    assert batch_eight["three_hundred_k"] == pytest.approx(hours["three_hundred_k"] * 8, rel=1e-3)
    autocast = variants["bf16_autocast"]
    assert autocast["measured_step_time_seconds"] == pytest.approx(
        step * autocast["speedup_factor"], rel=1e-3
    )
    assert hours["excludes"]
    mirror = protocol["profiling"]["gpu_hours"]
    for key in ("twenty_k", "hundred_k", "three_hundred_k"):
        assert mirror[key] == hours[key]


def test_protocol_manifest_hashes_are_recomputed_and_match_the_index() -> None:
    """The recorded hashes are verified against the bytes, not merely non-pending."""
    if not MANIFEST_DIR.is_dir():
        pytest.skip("frozen platform manifests are not present in this checkout")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    index = json.loads((MANIFEST_DIR / "n01-full-v1-index.json").read_text(encoding="utf-8"))
    data = protocol["data"]
    index_spec = data["manifest_index"]
    index_path = REPO_ROOT / index_spec["manifest_path"]
    assert preflight.sha256_file(index_path) == index_spec["sha256"]
    for platform, spec in data["platform_manifests"].items():
        manifest_path = REPO_ROOT / spec["manifest_path"]
        inventory_path = REPO_ROOT / spec["inventory_path"]
        evidence_path = REPO_ROOT / spec["loader_evidence_path"]
        assert preflight.sha256_file(manifest_path) == spec["manifest_sha256"], platform
        assert preflight.sha256_file(inventory_path) == spec["inventory_sha256"], platform
        assert preflight.sha256_file(evidence_path) == spec["loader_evidence_sha256"], platform
        declared = index["datasets"][platform]
        assert declared["manifest_sha256"] == spec["manifest_sha256"]
        assert declared["inventory_sha256"] == spec["inventory_sha256"]
        assert declared["loader_evidence_sha256"] == spec["loader_evidence_sha256"]
    for region in preflight.REQUIRED_SPLITS:
        for platform, spec in data["splits"][region]["platform_manifests"].items():
            assert spec["sha256"] == data["platform_manifests"][platform]["manifest_sha256"]


def test_protocol_agrees_with_workflow_a_config_module() -> None:
    from owac.frameworks.n01_full import config as n01_config

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert (
        tuple(protocol["exact_replay"]["unique_frames_choices"]) == n01_config.EXACT_WINDOW_CHOICES
    )
    assert tuple(protocol["input_identity"]["input_modes"]) == n01_config.INPUT_MODES
    assert protocol["input_identity"]["time_order"] == n01_config.InputIdentityConfig().time_order
    content = n01_config.ContentConfig()
    assert content.time_order_adapter == protocol["input_identity"]["time_order_adapter"]
    assert content.fusion_time_order == protocol["input_identity"]["fusion_time_order"]
    backbone = protocol["models_and_upstream"]["content_backbone"]
    assert (content.channels, content.num_blocks, content.feature_width) == (
        backbone["channels"],
        backbone["num_blocks"],
        backbone["feature_width"],
    )
    assert (content.bridge_in_dim, content.bridge_width) == (512, 2048)
    budget = n01_config.TrainingBudgetConfig()
    assert budget.model_seeds == tuple(protocol["seeds"]["model"]["values"])
    assert not set(protocol["seeds"]["model"]["values"]) & set(protocol["seeds"]["data"]["values"])
    assert (
        budget.validation_interval == protocol["validation_interval"]["updates_between_validations"]
    )


ALLOWED_HASH_SOURCES = {
    "recomputed_from_local_upstream_cache",
    "transcribed_source_manifest_json",
    "transcribed_from_local_provenance_record",
}


def _check_lock(lock: dict, cache_root: Path | None) -> dict:
    """Independent oracle for upstream_lock.json hash_source honesty.

    Works with the ignored cache present (recomputed mode) and absent
    (transcribed mode, e.g. inside a compliant source bundle).  It fails when a
    transcribed hash is claimed to have been recomputed, or the reverse.
    """
    violations: list[str] = []
    repositories = lock.get("repositories")
    if not isinstance(repositories, list) or not repositories:
        return {"ok": False, "mode": "unknown", "violations": ["repositories missing"]}
    cache_present = cache_root is not None and Path(cache_root).is_dir()
    recomputed_labels = 0
    transcribed_manifest_labels = 0
    recomputed_checked = 0
    for entry in repositories:
        name = entry.get("repository", "<unnamed>")
        commit = str(entry.get("commit", ""))
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            violations.append(f"{name}: commit is not a 40-hex pin")
        for field in ("license", "dependencies", "recipe", "weights_and_pretraining"):
            if not entry.get(field):
                violations.append(f"{name}: {field} missing")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            violations.append(f"{name}: files missing")
            continue
        for item in files:
            path = item.get("path")
            digest = str(item.get("sha256", ""))
            source = item.get("hash_source")
            if not path or len(digest) != 64:
                violations.append(f"{name}:{path}: malformed sha256/path")
                continue
            if source not in ALLOWED_HASH_SOURCES:
                violations.append(f"{name}:{path}: unknown hash_source {source!r}")
                continue
            if source == "recomputed_from_local_upstream_cache":
                recomputed_labels += 1
                if cache_present:
                    local = Path(cache_root) / name / path
                    if not local.is_file():
                        violations.append(
                            f"{name}:{path}: recomputed label, file absent from cache"
                        )
                    elif preflight.sha256_file(local) != digest:
                        violations.append(f"{name}:{path}: cache hash mismatch")
                    else:
                        recomputed_checked += 1
            elif source == "transcribed_source_manifest_json":
                transcribed_manifest_labels += 1
    verification = lock.get("hash_verification", {})
    matches = verification.get("recomputed_matches")
    declared_entries = verification.get("declared_hash_entries")
    if isinstance(matches, int) and matches != recomputed_labels:
        violations.append(
            f"hash_verification.recomputed_matches={matches} but "
            f"{recomputed_labels} recomputed labels"
        )
    if (
        isinstance(declared_entries, int)
        and recomputed_labels + transcribed_manifest_labels != declared_entries
    ):
        violations.append("declared_hash_entries != recomputed + transcribed labels")
    if cache_present and recomputed_checked == 0:
        violations.append("cache present but no declared file could be recomputed")
    if not cache_present:
        if not (lock.get("local_upstream_cache") or {}).get("present"):
            violations.append("transcribed mode without a lock-time cache record")
        if recomputed_labels and not verification.get("cache_absent_policy"):
            violations.append("recomputed labels without a documented cache-absent policy")
    return {
        "ok": not violations,
        "mode": "recomputed" if cache_present else "transcribed",
        "recomputed_labels": recomputed_labels,
        "transcribed_labels": transcribed_manifest_labels,
        "recomputed_checked": recomputed_checked,
        "violations": violations,
    }


def test_upstream_lock_is_honest_with_cache_and_without_it(tmp_path: Path) -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    cache = UPSTREAM_CACHE if UPSTREAM_CACHE.is_dir() else None
    present = _check_lock(lock, cache)
    assert present["ok"], present["violations"]
    assert present["mode"] == ("recomputed" if cache is not None else "transcribed")
    if cache is not None:
        assert present["recomputed_checked"] == 24
    # the same lock must also validate in a compliant snapshot that lacks the cache
    absent = _check_lock(lock, None)
    assert absent["ok"], absent["violations"]
    assert absent["mode"] == "transcribed"

    # discriminating assertion: mislabelling a recomputed hash as transcribed must fail
    tampered = json.loads(json.dumps(lock))
    first = tampered["repositories"][0]["files"][0]
    assert first["hash_source"] == "recomputed_from_local_upstream_cache"
    first["hash_source"] = "transcribed_source_manifest_json"
    result = _check_lock(tampered, cache)
    assert not result["ok"]
    assert any(
        "recomputed_matches" in item or "absent from cache" in item for item in result["violations"]
    )

    # a corrupted hash is caught whenever the bytes are actually available
    corrupted = json.loads(json.dumps(lock))
    corrupted["repositories"][0]["files"][0]["sha256"] = "0" * 64
    if cache is not None:
        result = _check_lock(corrupted, cache)
        assert not result["ok"]
        assert any("cache hash mismatch" in item for item in result["violations"])

    # claiming a lock-time recomputation without recording the cache environment fails
    no_record = json.loads(json.dumps(lock))
    no_record["local_upstream_cache"]["present"] = False
    result = _check_lock(no_record, None)
    assert not result["ok"]
    assert any("lock-time cache" in item for item in result["violations"])


def test_upstream_lock_matches_source_manifest_transcription() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(
        (REPO_ROOT / "docs/prompts/n01_v1_0/source_manifest.json").read_text(encoding="utf-8")
    )
    entries = {entry["repository"]: entry for entry in lock["repositories"]}
    manifest_repos = {entry["repository"]: entry for entry in manifest["repositories"]}
    assert set(manifest_repos) <= set(entries)
    assert len(entries) == 10  # nine upstream repositories plus the vendored DBSR pin
    for name, source in manifest_repos.items():
        entry = entries[name]
        assert entry["commit"] == source["commit"]
        declared = {item["path"]: item["sha256"] for item in source["files"] if item.get("sha256")}
        locked = {item["path"]: item["sha256"] for item in entry["files"]}
        assert locked == declared
    sea_raft = entries["princeton-vl/SEA-RAFT"]
    imagenet = sea_raft["recipe"]["imagenet_download_must_be_disabled"]
    assert "IMAGENET1K_V1" in imagenet["evidence"]
    assert "init_weight=False" in imagenet["requirement"]
    dbsr = entries["goutamgmb/deep-burst-sr"]
    assert dbsr["recipe"]["full_size_contract"]["channels"] == 64
    assert dbsr["recipe"]["time_order_contract"]["adapter"] == "oldest_first_to_newest_first"


def test_backend_runtime_assets_are_present() -> None:
    requirements = (BACKEND_DIR / "requirements-gpu.txt").read_text(encoding="utf-8")
    assert "cu128" in requirements and "torch==" in requirements
    readme = (BACKEND_DIR / "README.md").read_text(encoding="utf-8")
    assert "requirements-gpu.txt" in readme
    assert "CPU" in readme and "separate" in readme
    supervisor = (BACKEND_DIR / "owac-n01-v1.supervisor.conf").read_text(encoding="utf-8")
    assert "[program:owac-n01-v1-<stage>-<run>]" in supervisor
    assert "autostart=false" in supervisor


# ---------------------------------------------------------------------------------------
# bundle and verify
# ---------------------------------------------------------------------------------------


FROZEN_MANIFEST_FILES = {
    "data/manifests/n01-full-v1/n01-full-v1-index.json",
    "data/manifests/n01-full-v1/n01-full-v1-libero-manifest.json",
    "data/manifests/n01-full-v1/n01-full-v1-robotwin2-manifest.json",
}


def test_bundle_closure_includes_frozen_manifests_and_not_git_archive(tmp_path: Path) -> None:
    """The frozen manifests must ride in the bundle, and cannot come from git archive."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    root = tmp_path / "source"
    _build_mini_source(root)
    _git("init", "-q", cwd=root)
    _git("add", "src/owac/pkg/tracked_module.py", cwd=root)
    _git(
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "-m",
        "init",
        cwd=root,
    )
    output = tmp_path / "bundle"
    preflight.bundle(root, output, include_upstream=False)
    manifest = json.loads((output / "bundle-manifest.json").read_text(encoding="utf-8"))
    files = set(manifest["files"])
    assert FROZEN_MANIFEST_FILES <= files
    with tarfile.open(output / "source.tar.gz", "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    assert FROZEN_MANIFEST_FILES <= members

    # discriminating: git archive HEAD cannot produce this closure (the files are untracked)
    archived = subprocess.run(
        ["git", "archive", "HEAD"], cwd=root, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        tracked = {member.name for member in archive.getmembers() if member.isfile()}
    assert FROZEN_MANIFEST_FILES.isdisjoint(tracked)
    assert not any(name.startswith("data/manifests/") for name in tracked)


def test_bundle_closure_against_real_worktree(tmp_path: Path) -> None:
    """The real P0 bundle carries the manifests the protocol declares as frozen."""
    if not MANIFEST_DIR.is_dir():
        pytest.skip("frozen platform manifests are not present in this checkout")
    output = tmp_path / "bundle"
    preflight.bundle(REPO_ROOT, output, include_upstream=False)
    manifest = json.loads((output / "bundle-manifest.json").read_text(encoding="utf-8"))
    files = set(manifest["files"])
    assert FROZEN_MANIFEST_FILES <= files
    data_paths = sorted(path for path in files if path.startswith("data/manifests/n01-full-v1/"))
    assert len(data_paths) >= len(FROZEN_MANIFEST_FILES)
    report = preflight.verify_bundle(output, source_root=REPO_ROOT)
    assert report["failures"] == []
    closure = next(
        check for check in report["checks"] if check["id"] == "closure:data/manifests/n01-full-v1"
    )
    assert closure["ok"] is True


def _make_bundle(tmp_path: Path, *, with_git: bool) -> tuple[Path, Path]:
    root = tmp_path / "source"
    _build_mini_source(root)
    if with_git:
        _git("init", "-q", cwd=root)
        _git("add", "src/owac/pkg/tracked_module.py", cwd=root)
        _git(
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=test",
            "commit",
            "-q",
            "-m",
            "init",
            cwd=root,
        )
    (root / "src/owac/pkg/untracked_module.py").write_text("NEW = 1\n", encoding="utf-8")
    output = tmp_path / "bundle"
    preflight.bundle(root, output, include_upstream=False)
    return root, output


@pytest.mark.parametrize("with_git", [True, False])
def test_bundle_includes_untracked_and_drops_denied(tmp_path: Path, with_git: bool) -> None:
    if with_git and shutil.which("git") is None:
        pytest.skip("git is not available")
    root, output = _make_bundle(tmp_path, with_git=with_git)
    manifest = json.loads((output / "bundle-manifest.json").read_text(encoding="utf-8"))
    files = set(manifest["files"])
    assert "src/owac/pkg/untracked_module.py" in files
    assert "src/owac/pkg/tracked_module.py" in files
    assert "configs/encoders/n01_full/protocol_v1.json" in files
    assert "docs/prompts/n01_v1_0/source_manifest.json" in files
    for denied in (
        ".venv/lib/site.py",
        "outputs/run.json",
        ".env",
        "secrets.json",
        "src/owac/pkg/model.safetensors",
        "src/owac/pkg/__pycache__/x.pyc",
    ):
        assert denied not in files, denied
    assert manifest["exclusions"]["excluded_counts"]
    if with_git:
        assert "src/owac/pkg/untracked_module.py" in manifest["untracked_inclusion"]["included"]
        assert manifest["untracked_inclusion"]["missing"] == []
        assert "src/owac/pkg/model.safetensors" in manifest["untracked_inclusion"]["denied"]
        assert manifest["git"]["head"]

    with tarfile.open(output / "source.tar.gz", "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    assert names == files | {"bundle-manifest.json"}
    assert preflight.denied_reason("src/owac/pkg/untracked_module.py") is None


def test_verify_passes_then_fails_on_missing_mismatch_and_extra(tmp_path: Path) -> None:
    root, output = _make_bundle(tmp_path, with_git=False)
    report = preflight.verify_bundle(output, source_root=root)
    assert report["status"] in ("passed", "passed_pending")
    assert report["failures"] == []

    manifest_path = output / "bundle-manifest.json"
    original = manifest_path.read_text(encoding="utf-8")

    # 1. hash mismatch in the manifest
    manifest = json.loads(original)
    victim = sorted(manifest["files"])[0]
    manifest["files"][victim]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = preflight.verify_bundle(output, source_root=None)
    assert report["status"] == "failed"
    assert any(failure["id"] == "bundle:hashes" for failure in report["failures"])
    manifest_path.write_text(original, encoding="utf-8")

    # 2. a file dropped from the archive
    _rewrite_archive(output / "source.tar.gz", drop=victim)
    report = preflight.verify_bundle(output, source_root=None)
    assert report["status"] == "failed"
    assert any(failure["id"] == "bundle:no_missing_files" for failure in report["failures"])

    # 3. an injected extra file
    root, output = _make_bundle(tmp_path / "second", with_git=False)
    _rewrite_archive(output / "source.tar.gz", extra="src/owac/pkg/injected.py")
    report = preflight.verify_bundle(output, source_root=None)
    assert report["status"] == "failed"
    assert any(failure["id"] == "bundle:no_extra_files" for failure in report["failures"])

    # 4. a missing archive
    (output / "source.tar.gz").unlink()
    report = preflight.verify_bundle(output, source_root=None)
    assert report["status"] == "failed"
    assert any(failure["id"] == "archive:present" for failure in report["failures"])


def test_verify_detects_source_drift_and_new_untracked_files(tmp_path: Path) -> None:
    root, output = _make_bundle(tmp_path, with_git=False)
    assert preflight.verify_bundle(output, source_root=root)["failures"] == []

    (root / "src/owac/pkg/tracked_module.py").write_text("TRACKED = 2\n", encoding="utf-8")
    report = preflight.verify_bundle(output, source_root=root)
    drift = next(check for check in report["checks"] if check["id"] == "source_root:no_drift")
    assert drift["ok"] is False
    assert report["status"] == "failed"

    # A brand-new in-scope file that never entered the bundle is a closure failure.
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    root, output = _make_bundle(tmp_path / "git", with_git=True)
    (root / "src/owac/pkg/late_module.py").write_text("LATE = 1\n", encoding="utf-8")
    report = preflight.verify_bundle(output, source_root=root)
    closure = next(
        check for check in report["checks"] if check["id"] == "source_root:untracked_closure"
    )
    assert closure["ok"] is False
    assert "late_module.py" in closure["detail"]


def test_verify_cli_exit_code_and_report_file(tmp_path: Path) -> None:
    root, output = _make_bundle(tmp_path, with_git=False)
    report_path = tmp_path / "verify.json"
    exit_code = preflight.main(
        [
            "verify",
            "--bundle",
            str(output),
            "--source-root",
            str(root),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == preflight.EXIT_OK
    assert report_path.is_file()

    (output / "bundle-manifest.json").write_text("{}\n", encoding="utf-8")
    exit_code = preflight.main(["verify", "--bundle", str(output)])
    assert exit_code == preflight.EXIT_FAILED


def test_verify_refuses_unsafe_archive_member(tmp_path: Path) -> None:
    root, output = _make_bundle(tmp_path, with_git=False)
    _rewrite_archive(output / "source.tar.gz", extra="../escape.py")
    report = preflight.verify_bundle(output, source_root=None)
    assert report["status"] == "failed"
    assert any(failure["id"] == "archive:path_safety" for failure in report["failures"])

"""CLI tests for ``owac.evaluation.n02_full.p0``.

The upstream data/parity/profile modules do not exist yet, so these tests inject
small fake modules with ``sys.modules`` to exercise the CLI contract: missing
upstreams become ``not_run`` with a reason and exit code 0, a present upstream is
really called and its numbers are recorded, and the report layer derives its four
stage verdicts only from evidence files in the output tree. Every run uses a
temporary output directory; nothing here writes into the real ``outputs/`` tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from owac.evaluation.n02_full import p0

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_MODULE = "owac.frameworks.n02_full.data"
PARITY_MODULE = "owac.frameworks.n02_full.components.parity_suite"
PROFILE_MODULE = "owac.frameworks.n02_full.components.profile_targets"


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload: dict = {
        "runtime": {"dtype": "float32", "device": "cpu", "amp": False},
        "profile": {"updates": 5, "warmup_updates": 2, "batch_size": 2, "sequence_length": 8},
        "paths": {"output_root": str(tmp_path / "root")},
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                merged = dict(payload[key])
                merged.update(value)
                payload[key] = merged
            else:
                payload[key] = value
    target = tmp_path / "cfg.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# help / entry point
# ---------------------------------------------------------------------------


def test_help_lists_the_four_subcommands() -> None:
    text = p0.build_parser().format_help()
    for name in ("inventory", "parity", "profile", "report"):
        assert name in text
    assert p0.SUBCOMMANDS == ("inventory", "parity", "profile", "report")


def test_module_entry_point_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "owac.evaluation.n02_full.p0", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in ("inventory", "parity", "profile", "report"):
        assert name in result.stdout


def test_unknown_subcommand_is_refused() -> None:
    with pytest.raises(SystemExit) as excinfo:
        p0.build_parser().parse_args(["invent"])
    assert excinfo.value.code != 0


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def test_inventory_is_not_run_with_exit_zero_when_data_module_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, DATA_MODULE, None)
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "inventory"
    code = p0.main(["inventory", "--config", str(config), "--output", str(output)])
    assert code == 0
    payload = _load(output / "inventory.json")
    assert payload["status"] == "not_run"
    assert DATA_MODULE in payload["reason"]
    assert payload["dataset"] is None
    assert payload["interpreter"]["executable"] == sys.executable
    assert payload["device"] == "cpu"
    assert payload["config_hash"] == p0.n02_config.config_hash(p0.n02_config.load_config(config))


def test_inventory_calls_the_data_module_with_the_configured_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "ep0.npz").write_bytes(b"npz")
    calls: list[tuple] = []

    def audit_dataset_root(root, dataset_format):
        calls.append((root, dataset_format))
        return {"status": "ok", "num_episodes": 3, "data_hash": "abc123"}

    _fake_module(monkeypatch, DATA_MODULE, audit_dataset_root=audit_dataset_root)
    config = _write_config(
        tmp_path,
        {"data": {"cache_root": str(cache_root), "dataset_format": "libero"}},
    )
    output = tmp_path / "root" / "p0" / "inventory"
    assert p0.main(["inventory", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "inventory.json")
    assert payload["status"] == "ok"
    assert payload["dataset"] == {"status": "ok", "num_episodes": 3, "data_hash": "abc123"}
    assert calls == [(str(cache_root), "libero")]
    assert payload["git_sha"] == _git_head()


def test_inventory_does_not_call_the_data_module_for_an_empty_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "empty_cache"
    cache_root.mkdir()
    calls: list[str] = []
    _fake_module(
        monkeypatch,
        DATA_MODULE,
        audit_dataset_root=lambda root, dataset_format: calls.append(root),
    )
    config = _write_config(
        tmp_path, {"data": {"cache_root": str(cache_root), "dataset_format": "robotwin"}}
    )
    output = tmp_path / "root" / "p0" / "inventory"
    assert p0.main(["inventory", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "inventory.json")
    assert payload["status"] == "not_run"
    assert "empty" in payload["reason"]
    assert calls == []


def _inventory_fixture_config(tmp_path: Path, *, num_files: int = 3) -> Path:
    cache_root = tmp_path / "cache"
    cache_root.mkdir(exist_ok=True)
    for index in range(num_files):
        (cache_root / f"ep{index}.npz").write_bytes(b"npz")
    return _write_config(
        tmp_path, {"data": {"cache_root": str(cache_root), "dataset_format": "libero"}}
    )


def test_inventory_forwards_max_files_and_evidence_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []

    def audit_dataset_root(
        root, dataset_format, *, max_files=None, evidence_kind="unspecified_root"
    ):
        calls.append(
            {
                "root": root,
                "format": dataset_format,
                "max_files": max_files,
                "evidence_kind": evidence_kind,
            }
        )
        return {
            "status": "ok",
            "truncated": True,
            "files_checked": 2,
            "files_total": 1771,
            "evidence_kind": evidence_kind,
            "real_cache_evidence": evidence_kind == "real_cache",
        }

    _fake_module(monkeypatch, DATA_MODULE, audit_dataset_root=audit_dataset_root)
    config = _inventory_fixture_config(tmp_path, num_files=1)
    output = tmp_path / "root" / "p0" / "inventory"
    code = p0.main(
        [
            "inventory",
            "--config",
            str(config),
            "--output",
            str(output),
            "--max-files",
            "2",
            "--evidence-kind",
            "real_cache",
        ]
    )
    assert code == 0
    payload = _load(output / "inventory.json")
    assert calls == [
        {
            "root": str(tmp_path / "cache"),
            "format": "libero",
            "max_files": 2,
            "evidence_kind": "real_cache",
        }
    ]
    assert payload["audit_limits"] == {
        "max_files": 2,
        "evidence_kind": "real_cache",
        "evidence_kind_source": "upstream_report",
        "truncated": True,
        "truncated_source": "upstream_report",
        "files_checked": 2,
        "files_total": 1771,
        "audit_called": True,
    }
    assert payload["dataset"]["real_cache_evidence"] is True


def test_inventory_derives_the_truncation_flag_when_the_audit_does_not_report_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A minimal audit (no ``truncated``/``evidence_kind`` keys) must still record the cap."""
    calls: list[dict] = []

    def audit_dataset_root(
        root, dataset_format, *, max_files=None, evidence_kind="unspecified_root"
    ):
        calls.append({"max_files": max_files, "evidence_kind": evidence_kind})
        return {"status": "ok", "files_checked": 2, "files_total": 3}

    _fake_module(monkeypatch, DATA_MODULE, audit_dataset_root=audit_dataset_root)
    config = _inventory_fixture_config(tmp_path)
    output = tmp_path / "root" / "p0" / "inventory"
    assert (
        p0.main(["inventory", "--config", str(config), "--output", str(output), "--max-files", "2"])
        == 0
    )
    payload = _load(output / "inventory.json")
    assert calls == [{"max_files": 2, "evidence_kind": "unspecified_root"}]
    limits = payload["audit_limits"]
    assert limits["max_files"] == 2
    assert limits["truncated"] is True
    assert limits["truncated_source"] == "derived"
    assert limits["evidence_kind"] is None
    assert limits["evidence_kind_source"] == "default"
    assert limits["files_checked"] == 2
    assert limits["files_total"] == 3
    assert limits["audit_called"] is True


def test_inventory_without_limit_flags_keeps_the_upstream_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict] = []

    def audit_dataset_root(
        root, dataset_format, *, max_files=None, evidence_kind="unspecified_root"
    ):
        calls.append({"max_files": max_files, "evidence_kind": evidence_kind})
        return {
            "status": "ok",
            "truncated": False,
            "files_checked": 3,
            "files_total": 3,
            "evidence_kind": evidence_kind,
        }

    _fake_module(monkeypatch, DATA_MODULE, audit_dataset_root=audit_dataset_root)
    config = _inventory_fixture_config(tmp_path)
    output = tmp_path / "root" / "p0" / "inventory"
    assert p0.main(["inventory", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "inventory.json")
    # No keyword was sent, so the audit's own defaults applied.
    assert calls == [{"max_files": None, "evidence_kind": "unspecified_root"}]
    limits = payload["audit_limits"]
    assert limits["max_files"] is None
    assert limits["truncated"] is False
    assert limits["evidence_kind"] == "unspecified_root"
    assert limits["evidence_kind_source"] == "upstream_report"
    assert limits["files_checked"] == 3
    assert limits["audit_called"] is True


def test_inventory_records_audit_limits_even_when_the_audit_never_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "empty_cache"
    cache_root.mkdir()
    calls: list[str] = []
    _fake_module(
        monkeypatch,
        DATA_MODULE,
        audit_dataset_root=lambda *args, **kwargs: calls.append("called"),
    )
    config = _write_config(
        tmp_path, {"data": {"cache_root": str(cache_root), "dataset_format": "robotwin"}}
    )
    output = tmp_path / "root" / "p0" / "inventory"
    assert (
        p0.main(
            [
                "inventory",
                "--config",
                str(config),
                "--output",
                str(output),
                "--max-files",
                "2",
                "--evidence-kind",
                "real_cache",
            ]
        )
        == 0
    )
    payload = _load(output / "inventory.json")
    assert payload["status"] == "not_run"
    assert calls == []
    limits = payload["audit_limits"]
    assert limits["audit_called"] is False
    assert limits["max_files"] == 2
    assert limits["truncated"] is True
    assert limits["truncated_source"] == "derived"
    assert limits["evidence_kind"] == "real_cache"
    assert limits["evidence_kind_source"] == "requested"
    assert limits["files_checked"] is None


def test_inventory_records_a_failing_audit_as_not_run_with_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "ep0.npz").write_bytes(b"npz")

    def audit_dataset_root(root, dataset_format):
        raise RuntimeError("no readable episode")

    _fake_module(monkeypatch, DATA_MODULE, audit_dataset_root=audit_dataset_root)
    config = _write_config(
        tmp_path, {"data": {"cache_root": str(cache_root), "dataset_format": "libero"}}
    )
    output = tmp_path / "root" / "p0" / "inventory"
    assert p0.main(["inventory", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "inventory.json")
    assert payload["status"] == "not_run"
    assert "no readable episode" in payload["reason"]
    assert "RuntimeError" in payload["error_traceback"]


def test_inventory_reads_an_existing_runtime_probe(tmp_path: Path) -> None:
    probe_dir = tmp_path / "root" / "p0" / "runtime"
    probe_dir.mkdir(parents=True)
    probe = {"torch": "2.9.1+cu128", "cuda": True, "device": "RTX 5090"}
    (probe_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    config = _write_config(tmp_path)
    payload = p0.stage_inventory(
        p0.n02_config.load_config(config),
        p0.build_parser().parse_args(
            ["inventory", "--config", str(config), "--output", str(tmp_path / "out")]
        ),
    )
    assert payload["runtime_probe"]["available"] is True
    assert payload["runtime_probe"]["payload"] == probe
    assert payload["runtime_probe"]["file_sha256"] == p0.sha256_file(probe_dir / "probe.json")


def test_lock_summary_is_a_summary_not_the_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "third_party.lock.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"id": "r01", "commit": "abc", "sha256": "1" * 64},
                    {"id": "r09", "commit": "def", "file_sha256": "2" * 64, "license": "MIT"},
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = p0._lock_summary(path)
    assert summary["available"] is True
    assert summary["entry_count"] == 2
    assert summary["file_sha256"] == p0.sha256_file(path)
    assert len(summary["canonical_sha256"]) == 64
    assert [entry["id"] for entry in summary["entries"]] == ["r01", "r09"]
    assert summary["entries_truncated"] is False

    big = tmp_path / "big.lock.json"
    big.write_text(json.dumps([{"id": f"r{i}"} for i in range(60)]), encoding="utf-8")
    assert p0._lock_summary(big)["entries_truncated"] is True

    missing = p0._lock_summary(tmp_path / "absent.lock.json")
    assert missing["available"] is False and missing["reason"]
    broken = tmp_path / "broken.lock.json"
    broken.write_text("{oops", encoding="utf-8")
    assert p0._lock_summary(broken)["available"] is False


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def test_parity_writes_the_suite_dict_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict = {}

    def run_all(config, device="cpu"):
        received["config"] = config
        received["device"] = device
        return {
            "status": "ok",
            "numerical": "pass",
            "checks_total": 4,
            "checks_passed": 4,
            "custom": {"max_abs_error": 1e-6},
        }

    _fake_module(monkeypatch, PARITY_MODULE, run_all=run_all)
    config = _write_config(tmp_path, {"worlds": 4})
    output = tmp_path / "root" / "p0" / "parity"
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["custom"] == {"max_abs_error": 1e-6}
    assert payload["checks_total"] == 4
    assert payload["numerical"] == "pass"
    assert payload["status"] == "ok"
    assert payload["stage"] == "parity"
    assert payload["config_hash"] == p0.n02_config.config_hash(received["config"])
    assert received["config"].worlds == 4
    assert received["device"] == "cpu"


def test_parity_forwards_the_device_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    received: dict = {}
    _fake_module(
        monkeypatch,
        PARITY_MODULE,
        run_all=lambda config, device="cpu": received.update(device=device) or {"status": "ok"},
    )
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "parity"
    assert (
        p0.main(["parity", "--config", str(config), "--output", str(output), "--device", "cuda:1"])
        == 0
    )
    assert received["device"] == "cuda:1"


def test_parity_is_not_run_when_the_suite_is_missing_or_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "parity"

    monkeypatch.setitem(sys.modules, PARITY_MODULE, None)
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == "not_run" and PARITY_MODULE in payload["reason"]

    def run_all(config, device="cpu"):
        raise ValueError("upstream op unavailable")

    _fake_module(monkeypatch, PARITY_MODULE, run_all=run_all)
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == "not_run"
    assert "upstream op unavailable" in payload["reason"]

    _fake_module(monkeypatch, PARITY_MODULE, run_all=lambda config, device="cpu": "pass")
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == "not_run"
    assert "must return a dict" in payload["reason"]

    _fake_module(
        monkeypatch,
        PARITY_MODULE,
        run_all=lambda config, device="cpu": {"status": "not_run", "reason": "no reference env"},
    )
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == "not_run"
    assert payload["reason"] == "no reference env"


@pytest.mark.parametrize("status", ["pass", "fail", "not_run", "ok", "incomplete"])
def test_parity_preserves_the_suite_top_level_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    """The suite's own status vocabulary (including "pass") must survive verbatim."""
    _fake_module(
        monkeypatch,
        PARITY_MODULE,
        run_all=lambda config, device="cpu": {
            "status": status,
            "pass_count": 11,
            "not_run_count": 5,
        },
    )
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "parity"
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == status
    assert payload["pass_count"] == 11
    assert payload["not_run_count"] == 5
    assert "unrecognized" not in str(payload.get("reason", ""))


def test_parity_rewrites_only_an_unknown_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_module(
        monkeypatch, PARITY_MODULE, run_all=lambda config, device="cpu": {"status": "all_good"}
    )
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "parity"
    assert p0.main(["parity", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "parity.json")
    assert payload["status"] == "incomplete"
    assert "unrecognized" in payload["reason"]


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


class _FakeTarget:
    """Test double for one profile target; it really executes update calls."""

    def __init__(
        self,
        name: str,
        *,
        params: int | None = 1234,
        batch: int | None = 2,
        sequence: int | None = 8,
        frames: int | None = None,
        parts: tuple = (),
    ) -> None:
        self.name = name
        self.calls = 0
        self.parameter_count = params
        self.batch_size = batch
        self.sequence_length = sequence
        self.composed_parts = parts
        if frames is not None:
            self.frames_per_update = frames

    def update(self) -> None:
        self.calls += 1


def test_profile_measures_real_updates_and_separates_three_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    alpha_base = _FakeTarget("alpha_base", params=1234, frames=16)
    beta = _FakeTarget("beta", params=99, batch=4, sequence=4)

    class _Composed:
        name = "alpha"
        composed_parts = ("alpha_base", "beta")

    class _NoUpdate:
        name = "broken"

    targets = [alpha_base, beta, _Composed(), _NoUpdate()]
    _fake_module(monkeypatch, PROFILE_MODULE, build_profile_targets=lambda config: targets)
    config = _write_config(
        tmp_path,
        {"profile": {"updates": 5, "warmup_updates": 2, "batch_size": 2, "sequence_length": 8}},
    )
    output = tmp_path / "root" / "p0" / "profile"
    assert p0.main(["profile", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "profile.json")

    assert set(payload) >= {"measured", "composed_estimate", "not_measured"}
    assert payload["status"] == "incomplete"
    assert payload["protocol"] == {
        "updates": 5,
        "warmup_updates": 2,
        "dtype": "float32",
        "amp": False,
    }
    assert alpha_base.calls == 5 + 2
    assert beta.calls == 5 + 2

    measured = {entry["name"]: entry for entry in payload["measured"]["targets"]}
    assert set(measured) == {"alpha_base", "beta"}
    for entry in measured.values():
        assert entry["updates"] == 5
        assert entry["seconds_per_update"]["mean"] > 0.0
        timing = entry["seconds_per_update"]
        assert timing["min"] <= timing["p50"] <= timing["max"]
        assert timing["min"] <= timing["p95"] <= timing["max"]
        assert entry["seconds_total"] > 0.0
        assert entry["device_name"] == "cpu"
        assert entry["peak_memory"]["allocated_bytes"] is None
        assert "not measured" in entry["peak_memory"]["source"]
    assert measured["alpha_base"]["parameter_count"] == 1234
    assert measured["alpha_base"]["throughput_basis"] == "declared frames_per_update"
    assert measured["alpha_base"]["throughput_unit"] == "samples"
    assert measured["alpha_base"]["throughput_per_second"] > 0.0
    assert measured["beta"]["throughput_basis"] == "batch_size*sequence_length"
    assert measured["beta"]["effective_batch_size"] == 4
    assert measured["beta"]["sequence_length"] == 4
    assert measured["beta"]["parameter_count"] == 99
    assert payload["runtime"]["torch_version"] == torch.__version__
    assert payload["runtime"]["device_name"] == "cpu"
    assert payload["runtime"]["device_type"] == "cpu"
    assert payload["runtime"]["cuda_queried"] is False
    assert payload["runtime"]["cuda_available"] is None
    assert payload["runtime"]["reason"]

    composed = payload["composed_estimate"]
    assert composed["available"] is True
    estimate = composed["targets"][0]
    assert estimate["name"] == "alpha"
    assert estimate["status"] == "incomplete"
    assert estimate["basis"] == ["alpha_base", "beta"]
    expected = (
        measured["alpha_base"]["seconds_per_update"]["mean"]
        + measured["beta"]["seconds_per_update"]["mean"]
    )
    assert estimate["estimated_seconds_per_update"] == pytest.approx(expected)
    assert "composed estimate" in estimate["caveat"]

    failed = payload["not_measured"]["targets"]
    assert [entry["name"] for entry in failed] == ["alpha", "broken"]
    assert "update" in failed[-1]["reason"]
    items = payload["not_measured"]["items"]
    names = [item["name"] for item in items]
    assert "peak_memory_allocated_bytes" in names
    assert all(item["reason"] for item in items)


def test_profile_honours_max_updates_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _FakeTarget("alpha")
    _fake_module(monkeypatch, PROFILE_MODULE, build_profile_targets=lambda config: [target])
    config = _write_config(tmp_path, {"profile": {"updates": 100, "warmup_updates": 3}})
    output = tmp_path / "root" / "p0" / "profile"
    assert (
        p0.main(
            [
                "profile",
                "--config",
                str(config),
                "--output",
                str(output),
                "--max-updates",
                "3",
            ]
        )
        == 0
    )
    payload = _load(output / "profile.json")
    assert payload["protocol"]["updates"] == 3
    assert payload["protocol"]["warmup_updates"] == 3
    assert target.calls == 3 + 3
    assert payload["measured"]["targets"][0]["updates"] == 3
    assert payload["status"] == "ok"


def test_profile_is_not_run_when_targets_are_missing_or_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "profile"

    monkeypatch.setitem(sys.modules, PROFILE_MODULE, None)
    assert p0.main(["profile", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "profile.json")
    assert payload["status"] == "not_run" and PROFILE_MODULE in payload["reason"]
    assert payload["measured"]["targets"] == []
    assert payload["not_measured"]["items"]

    def build(config):
        raise RuntimeError("no GPU runtime")

    _fake_module(monkeypatch, PROFILE_MODULE, build_profile_targets=build)
    assert p0.main(["profile", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "profile.json")
    assert payload["status"] == "not_run" and "no GPU runtime" in payload["reason"]

    _fake_module(monkeypatch, PROFILE_MODULE, build_profile_targets=lambda config: [])
    assert p0.main(["profile", "--config", str(config), "--output", str(output)]) == 0
    payload = _load(output / "profile.json")
    assert payload["status"] == "not_run" and "no targets" in payload["reason"]


# ---------------------------------------------------------------------------
# device probes: a cpu device must never touch a CUDA API
# ---------------------------------------------------------------------------


def _cuda_stub(calls: list, label: str, *, raises: bool = True):
    """Stub that records the call; by default it raises so a call cannot pass."""

    def stub(*args, **kwargs):
        calls.append((label, args))
        if raises:
            raise AssertionError(f"torch.cuda.{label} must not be called for this device")
        return None

    return stub


_CUDA_API_NAMES = (
    "is_available",
    "get_device_name",
    "reset_peak_memory_stats",
    "max_memory_allocated",
    "max_memory_reserved",
)


def test_runtime_info_with_device_cpu_never_touches_cuda_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU logic case: no GPU required, and every CUDA entry point is booby-trapped."""
    torch = pytest.importorskip("torch")
    calls: list = []
    for name in _CUDA_API_NAMES:
        monkeypatch.setattr(torch.cuda, name, _cuda_stub(calls, name))

    info = p0._runtime_info("cpu")

    assert calls == []
    assert info["device"] == "cpu"
    assert info["device_type"] == "cpu"
    assert info["device_name"] == "cpu"
    assert info["cuda_queried"] is False
    assert info["cuda_available"] is None
    assert info["torch_version"] == torch.__version__
    assert info["reason"]


def test_runtime_info_queries_cuda_only_for_a_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cuda branch really does query; the cpu branch above never does."""
    torch = pytest.importorskip("torch")
    calls: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda device=None: calls.append(device) or "Stub 5090"
    )

    cuda_info = p0._runtime_info("cuda:1")
    assert cuda_info["device_type"] == "cuda"
    assert cuda_info["cuda_queried"] is True
    assert cuda_info["cuda_available"] is True
    assert cuda_info["device_name"] == "Stub 5090"
    assert calls == ["cuda:1"]

    cpu_info = p0._runtime_info("cpu")
    assert cpu_info["device_name"] == "cpu"
    assert calls == ["cuda:1"]  # the cpu probe added no CUDA call


def test_runtime_info_cuda_device_without_cuda_records_a_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", _cuda_stub(calls, "get_device_name"))

    info = p0._runtime_info("cuda")
    assert info["cuda_queried"] is True
    assert info["cuda_available"] is False
    assert info["device_name"] is None
    assert "is_available" in info["reason"]
    assert calls == []


def test_profile_with_device_cpu_is_cuda_free_even_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: a CUDA host asked for --device cpu must not reach any CUDA API."""
    torch = pytest.importorskip("torch")
    calls: list = []

    def fake_is_available() -> bool:
        calls.append("is_available")
        return True

    monkeypatch.setattr(torch.cuda, "is_available", fake_is_available)
    for name in _CUDA_API_NAMES[1:]:
        monkeypatch.setattr(torch.cuda, name, _cuda_stub(calls, name))

    target = _FakeTarget("alpha")
    _fake_module(monkeypatch, PROFILE_MODULE, build_profile_targets=lambda config: [target])
    config = _write_config(tmp_path, {"profile": {"updates": 2, "warmup_updates": 1}})
    output = tmp_path / "root" / "p0" / "profile"
    code = p0.main(["profile", "--config", str(config), "--output", str(output), "--device", "cpu"])
    assert code == 0
    payload = _load(output / "profile.json")

    assert payload["status"] == "ok"
    assert calls == []  # not even the capability query ran
    runtime = payload["runtime"]
    assert runtime["device_name"] == "cpu"
    assert runtime["cuda_queried"] is False
    assert runtime["cuda_available"] is None
    entry = payload["measured"]["targets"][0]
    assert entry["device_name"] == "cpu"
    assert entry["peak_memory"]["allocated_bytes"] is None
    assert entry["peak_memory"]["reserved_bytes"] is None
    assert "not measured" in entry["peak_memory"]["source"]
    item = next(
        entry
        for entry in payload["not_measured"]["items"]
        if entry["name"] == "peak_memory_allocated_bytes"
    )
    assert "no CUDA API was queried" in item["reason"]
    assert target.calls == 2 + 1


class _ModuleLikeTarget:
    """Minimal nn.Module look-alike: declared profile_meta drives its input."""

    def __init__(self, meta: dict) -> None:
        self.profile_meta = meta
        self.seen_shapes: list = []

    def forward(self, imgs, mask_ratio=None):  # noqa: ARG002 - the signature is the point
        self.seen_shapes.append(tuple(imgs.shape))
        return imgs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _ExtraRequiredTarget(_ModuleLikeTarget):
    def forward(self, imgs, mystery):  # noqa: ARG002 - required, undescribed parameter
        self.seen_shapes.append(tuple(imgs.shape))
        return imgs


def test_profile_target_inputs_come_from_profile_meta() -> None:
    torch = pytest.importorskip("torch")
    meta = {
        "batch_size": 2,
        "sequence_length": None,
        "input_shape": (2, 4, 16, 16),
        "params": 12345,
        "notes": "declared by a module-like target",
    }
    raw = _ModuleLikeTarget(meta)
    adapter = p0._ProfileTarget(raw, 0, "declared_target")

    args, kwargs, reason = p0._target_inputs(torch, adapter, "cpu")
    assert reason is None
    assert kwargs == {}
    assert len(args) == 1 and tuple(args[0].shape) == (2, 4, 16, 16)
    assert adapter.parameter_count() == 12345
    assert adapter.int_field("batch_size") == 2

    measured, failure = p0._measure_target(
        adapter, "cpu", 2, 0, {"cuda_queried": False, "reason": "cpu run"}
    )
    assert failure is None
    assert measured is not None
    # forward(imgs, mask_ratio=None): the optional parameter keeps its default
    assert raw.seen_shapes == [(2, 4, 16, 16)] * 2
    assert measured["parameter_count"] == 12345
    assert measured["effective_batch_size"] == 2
    assert measured["sequence_length"] is None
    assert measured["throughput_per_second"] is not None
    assert measured["throughput_basis"] == "batch_size (target declares no time axis)"
    assert measured["notes"] == ["declared by a module-like target"]
    assert measured["name"] == "declared_target"


def test_profile_target_with_an_undescribed_required_parameter_is_not_measured() -> None:
    torch = pytest.importorskip("torch")
    raw = _ExtraRequiredTarget({"batch_size": 1, "input_shape": (1, 4, 8, 8), "params": 3})
    adapter = p0._ProfileTarget(raw, 0, "mystery_target")
    args, kwargs, reason = p0._target_inputs(torch, adapter, "cpu")
    assert (args, kwargs) == ([], {})
    assert reason is not None and "mystery" in reason

    measured, failure = p0._measure_target(
        adapter, "cpu", 1, 0, {"cuda_queried": False, "reason": "cpu run"}
    )
    assert measured is None
    assert failure is not None
    assert failure["name"] == "mystery_target"
    assert "mystery" in failure["reason"]


def test_measure_target_reads_cuda_peak_memory_only_when_cuda_was_queried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Branch test with stubbed counters: the cuda memory path is reachable and ordered."""
    torch = pytest.importorskip("torch")
    calls: list = []
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append(("reset", device))
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device=None: (calls.append(("allocated", device)), 4096)[1],
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device=None: (calls.append(("reserved", device)), 8192)[1],
    )

    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: calls.append(("sync", device))
    )
    target = _FakeTarget("alpha")
    adapter = p0._ProfileTarget(target, 0)
    measured, failure = p0._measure_target(
        adapter, "cuda:0", 2, 1, {"cuda_queried": True, "device_name": "Stub"}
    )
    assert failure is None
    assert measured is not None
    assert measured["peak_memory"]["allocated_bytes"] == 4096
    assert measured["peak_memory"]["reserved_bytes"] == 8192
    assert measured["peak_memory"]["source"] == "torch.cuda.max_memory_allocated/reserved"
    assert [label for label, _ in calls] == [
        "sync",
        "reset",
        "sync",
        "sync",
        "sync",
        "sync",
        "allocated",
        "reserved",
    ]
    assert target.calls == 2 + 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _write_evidence(root: Path, stage: str, payload: dict) -> Path:
    path = root / "p0" / stage / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_composes_all_four_documents_from_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(p0, "DEFAULT_OUTPUT_ROOT", str(tmp_path / "isolated"))
    root = tmp_path / "root"
    inventory = _write_evidence(
        root, "inventory", {"status": "ok", "num_episodes": 12, "data_hash": "deadbeef"}
    )
    parity = _write_evidence(
        root,
        "parity",
        {"status": "ok", "numerical": "pass", "checks_total": 6, "max_abs_error": 1e-6},
    )
    profile = _write_evidence(
        root,
        "profile",
        {
            "status": "ok",
            "protocol": {"updates": 200, "warmup_updates": 20, "dtype": "float32"},
            "measured": {
                "targets": [
                    {
                        "name": name,
                        "operation": "train_step",
                        "updates": 200,
                        "optimizer_steps": 220,
                        "timing_synchronized": True,
                        "parameter_count": 28_000_000,
                        "seconds_per_update": {"mean": 0.01, "p95": 0.02},
                        "throughput_per_second": 200.0,
                        "peak_memory": {"allocated_bytes": None, "reserved_bytes": None},
                    }
                    for name in ("convnext_raw4", "convnext_raw3", "fcmae_full", "savi_512")
                ]
            },
        },
    )
    config = _write_config(tmp_path)
    output = root / "p0" / "report"
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0

    decision = _load(output / "stage_decision.json")
    assert decision["implementation"] == "pass"
    assert decision["numerical"] == "pass"
    assert decision["behavior"] == "not_run"
    assert decision["closed_loop"] == "not_run"
    assert decision["git_sha"] == _git_head()
    assert decision["config_hash"] == p0.n02_config.config_hash(p0.n02_config.load_config(config))
    assert decision["data_hash"] == "deadbeef"
    for path in (inventory, parity, profile):
        assert str(path) in decision["evidence_paths"]

    report = _load(output / "report.json")
    assert report["status"] == "ok"
    assert report["data_hash"] == "deadbeef"
    assert report["git_sha"] == _git_head()
    assert report["evidence"]["parity"]["status"] == "ok"
    assert report["evidence"]["profile"]["sha256"] == p0.sha256_file(profile)

    metrics = _load(output / "metrics.json")
    assert metrics["inventory"]["num_episodes"] == 12
    assert metrics["parity"]["max_abs_error"] == 1e-6
    assert metrics["profile"]["updates"] == 200
    assert metrics["profile"]["targets"][0]["parameter_count"] == 28_000_000
    assert metrics["profile"]["targets"][0]["mean_seconds_per_update"] == 0.01

    manifest = _load(output / "run_manifest.json")
    assert manifest["git_sha"] == _git_head()
    assert manifest["config_hash"] == report["config_hash"]
    assert manifest["argv"]
    assert [item["path"] for item in manifest["artifacts"]][-1].endswith("report.json")
    assert all(item["sha256"] for item in manifest["artifacts"])


def test_report_treats_a_passing_parity_status_as_successful_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A parity suite verdict of "pass" is success, not an unrecognized status."""
    monkeypatch.setattr(p0, "DEFAULT_OUTPUT_ROOT", str(tmp_path / "isolated"))
    root = tmp_path / "root"
    _write_evidence(root, "inventory", {"status": "ok", "num_files": 3})
    _write_evidence(
        root,
        "parity",
        {"status": "pass", "numerical": "pass", "pass_count": 11, "not_run_count": 5},
    )
    _write_evidence(
        root,
        "profile",
        {
            "status": "ok",
            "protocol": {"updates": 200, "warmup_updates": 20},
            "measured": {"targets": []},
        },
    )
    config = _write_config(tmp_path)
    output = root / "p0" / "report"
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0

    report = _load(output / "report.json")
    assert report["status"] == "incomplete"
    assert "not successful" not in report["reason"]
    decision = _load(output / "stage_decision.json")
    assert decision["implementation"] == "incomplete"
    assert decision["numerical"] == "incomplete"
    assert decision["behavior"] == "not_run"
    assert decision["closed_loop"] == "not_run"
    metrics = _load(output / "metrics.json")
    assert metrics["parity"]["status"] == "pass"
    assert metrics["parity"]["pass_count"] == 11
    assert metrics["parity"]["not_run_count"] == 5


def test_report_marks_missing_evidence_not_run_and_never_invents_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(p0, "DEFAULT_OUTPUT_ROOT", str(tmp_path / "isolated"))
    config = _write_config(tmp_path)
    output = tmp_path / "root" / "p0" / "report"
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    decision = _load(output / "stage_decision.json")
    assert {
        decision[axis] for axis in ("implementation", "numerical", "behavior", "closed_loop")
    } == {"not_run"}
    assert _load(output / "report.json")["status"] == "not_run"
    assert _load(output / "metrics.json")["profile"]["targets_measured"] == 0

    root = tmp_path / "other"
    _write_evidence(root, "inventory", {"status": "ok", "num_files": 2})
    _write_evidence(
        root, "parity", {"status": "ok", "numerical": "fail", "reason": "1e-3 mismatch"}
    )
    _write_evidence(root, "profile", {"status": "not_run", "reason": "no targets"})
    output = root / "p0" / "report"
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    decision = _load(output / "stage_decision.json")
    assert decision["numerical"] == "fail"
    assert decision["implementation"] == "fail"
    assert decision["behavior"] == "not_run"
    assert decision["closed_loop"] == "not_run"
    assert decision["data_hash"] is None
    assert _load(output / "report.json")["status"] == "incomplete"


def test_report_derives_the_numerical_verdict_from_the_suite_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no explicit ``numerical`` field the suite's status is the verdict."""
    monkeypatch.setattr(p0, "DEFAULT_OUTPUT_ROOT", str(tmp_path / "isolated"))
    root = tmp_path / "root"
    _write_evidence(root, "inventory", {"status": "not_run", "reason": "no data root"})
    _write_evidence(
        root,
        "profile",
        {"status": "ok", "protocol": {"updates": 200}, "measured": {"targets": []}},
    )
    config = _write_config(tmp_path)
    output = root / "p0" / "report"

    _write_evidence(
        root, "parity", {"status": "pass", "pass_count": 11, "fail_count": 0, "not_run_count": 5}
    )
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    decision = _load(output / "stage_decision.json")
    assert decision["numerical"] == "incomplete"
    assert decision["implementation"] == "incomplete"
    assert any("not_run_count=5" in note for note in decision["notes"])

    _write_evidence(root, "parity", {"status": "fail", "pass_count": 3, "fail_count": 8})
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    decision = _load(output / "stage_decision.json")
    assert decision["numerical"] == "fail"

    _write_evidence(root, "parity", {"status": "not_run", "reason": "no reference environment"})
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    decision = _load(output / "stage_decision.json")
    assert decision["numerical"] == "not_run"
    assert any("no reference environment" in note for note in decision["notes"])


def test_report_reads_a_stage_specific_reason_for_not_run_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(p0, "DEFAULT_OUTPUT_ROOT", str(tmp_path / "isolated"))
    root = tmp_path / "root"
    _write_evidence(root, "inventory", {"status": "not_run", "reason": "cache_root missing"})
    config = _write_config(tmp_path)
    output = root / "p0" / "report"
    assert p0.main(["report", "--config", str(config), "--output", str(output)]) == 0
    report = _load(output / "report.json")
    assert report["evidence"]["inventory"]["status"] == "not_run"
    assert report["evidence"]["inventory"]["reason"] == "cache_root missing"
    assert report["evidence"]["parity"]["path"] is None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_write_json_is_canonical_and_creates_parents(tmp_path: Path) -> None:
    path = p0.write_json(tmp_path / "nested" / "deep" / "out.json", {"b": 1, "a": "中文"})
    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')
    assert '"中文"' in text
    assert text.endswith("\n")
    assert text.splitlines()[1].startswith("  ")
    assert json.loads(text) == {"a": "中文", "b": 1}


def test_git_sha_matches_git_rev_parse() -> None:
    sha = p0.git_sha()
    assert sha == _git_head()
    assert sha is None or len(sha) == 40


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

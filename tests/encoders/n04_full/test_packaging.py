"""Packaging contract for the N04 source snapshot tool.

The tests build real throwaway git repositories (never the OWAC checkout) so the
tool's guarantees are exercised against actual git output: current working-tree
bytes, explicit untracked inclusion, exclusions, determinism and remote
verification.  The last test renders the N04 supervisor template and checks the
lifecycle/isolation contract it encodes.
"""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "n04_full" / "package_source.py"
TEMPLATE_PATH = REPO_ROOT / "scripts" / "n04_full" / "n04_supervisor.conf.template"
SNAPSHOT_RE = re.compile(r"^n04-s0-\d{8}T\d{6}Z-[0-9a-f]{12}$")
FIXED_TS = "20261014T050000Z"
PLACEHOLDERS = {
    "__STAGE__": "s1",
    "__RUN_ID__": "demo-1",
    "__SNAPSHOT_ID__": "n04-s0-20261014T035852Z-906f1376bda2",
    "__SOURCE_DIR__": "/workspace/owac/research/n04-v1/source/n04-s0-20261014T035852Z-906f1376bda2",
    "__RUN_DIR__": "/workspace/owac/research/n04-v1/runs/s1/demo-1",
    "__PYTHON__": "/workspace/owac/research/n04-v1/envs/encoder/bin/python",
    "__ENTRY_MODULE__": "owac.training.n04_full",
    "__CONFIG__": "/workspace/owac/research/n04-v1/source/s1-demo/configs/n04/full/s1.json",
    "__CUDA_DEVICE__": "0",
    "__EXTRA_ARGS__": "",
}


@pytest.fixture(scope="module")
def packer():
    spec = importlib.util.spec_from_file_location("n04_package_source", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> bytes:
    env = dict(
        os.environ,
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, env=env)
    return proc.stdout


def _new_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "owac").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("root readme\n", encoding="utf-8")
    (repo / "src" / "owac" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "test_mod.py").write_text("def test_ok():\n    assert True\n", "utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, env=dict(os.environ))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(packer, repo: Path, out_dir: Path, *extra: str) -> int:
    return packer.main(
        [
            "build",
            "--repo-root",
            str(repo),
            "--output-dir",
            str(out_dir),
            "--timestamp",
            FIXED_TS,
            *extra,
        ]
    )


def _extract(package_path: Path, dest: Path) -> None:
    with tarfile.open(package_path) as archive:
        try:
            archive.extractall(dest, filter="data")
        except TypeError:  # Python 3.11 without the extraction-filter backport
            archive.extractall(dest)


def _load_manifest(out_dir: Path) -> dict:
    manifests = sorted(out_dir.glob("*.manifest.json"))
    assert len(manifests) == 1, manifests
    return json.loads(manifests[0].read_text("utf-8"))


def test_build_packs_working_tree_bytes_and_proves_uncommitted_content(
    packer, tmp_path: Path
) -> None:
    repo = _new_repo(tmp_path)
    (repo / "src" / "owac" / "mod.py").write_text("VALUE = 2  # edited\n", encoding="utf-8")
    (repo / "src" / "owac" / "untracked_new.py").write_text("NEW = True\n", encoding="utf-8")
    (repo / "configs").mkdir()
    (repo / "configs" / "run.json").write_text('{"seed": 1}\n', encoding="utf-8")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "artifact.py").write_text("generated\n", encoding="utf-8")
    (repo / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv" / "lib" / "vendored.py").write_text("vendored\n", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "blob.npz").write_bytes(b"\x00" * 64)
    (repo / "docs").mkdir()
    (repo / "docs" / "fig.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    _git(repo, "add", "docs/fig.png", "README.md")
    _git(repo, "commit", "-q", "-m", "tracked figure")

    out_dir = tmp_path / "packages"
    assert _build(packer, repo, out_dir) == 0
    manifest = _load_manifest(out_dir)

    assert SNAPSHOT_RE.match(manifest["snapshot_id"])
    package_path = out_dir / f"{manifest['snapshot_id']}.tar.gz"
    assert _sha256(package_path) == manifest["package"]["sha256"]
    assert manifest["snapshot_id"].endswith(manifest["package"]["sha256"][:12])
    assert manifest["git"]["head"] == _git(repo, "rev-parse", "HEAD").decode().strip()
    assert (
        manifest["git"]["tracked_diff_sha256"]
        == hashlib.sha256(_git(repo, "diff", "--binary", "HEAD")).hexdigest()
    )

    packed = {record["path"]: record for record in manifest["files"]}
    assert packed["src/owac/mod.py"]["sha256"] == _sha256(repo / "src" / "owac" / "mod.py")
    assert packed["src/owac/mod.py"]["origin"] == "tracked"
    assert packed["src/owac/untracked_new.py"]["origin"] == "untracked"
    assert packed["configs/run.json"]["sha256"] == _sha256(repo / "configs" / "run.json")
    assert packed["docs/fig.png"]["sha256"] == _sha256(repo / "docs" / "fig.png")
    assert "docs/fig.png" in {item["path"] for item in manifest["warnings"]}

    for absent in ("outputs/artifact.py", ".venv/lib/vendored.py", "data/blob.npz"):
        assert absent not in packed
    reasons = {item["path"]: item["reason"] for item in manifest["excluded"]}
    assert reasons["outputs/artifact.py"] == "deny-dir:outputs"
    assert reasons[".venv/lib/vendored.py"] == "deny-dir:.venv"
    assert reasons["data/blob.npz"] == "untracked-out-of-scope:data"

    evidence = manifest["uncommitted_evidence"]
    assert evidence["verified"] is True
    assert evidence["sample"]["present_in_head"] is False
    assert evidence["sample"]["present_in_manifest"] is True
    assert evidence["sample"]["sha256"] == packed[evidence["sample"]["path"]]["sha256"]
    assert "src/owac/untracked_new.py" in evidence["untracked_not_in_head"]
    assert "src/owac/mod.py" in evidence["modified_tracked"]
    assert evidence["sample"]["path"] == "configs/run.json"

    # the manifest hash must describe the bytes stored in the archive
    with tarfile.open(package_path) as archive:
        members = {info.name: info for info in archive.getmembers()}
        assert set(members) == set(packed)
        for name, info in members.items():
            data = archive.extractfile(info).read()
            assert hashlib.sha256(data).hexdigest() == packed[name]["sha256"]
            assert len(data) == packed[name]["bytes"]
        extracted = archive.extractfile(members["src/owac/untracked_new.py"]).read()
    assert extracted == (repo / "src" / "owac" / "untracked_new.py").read_bytes()
    assert manifest["stability"] == {
        "stable": True,
        "unstable_files": [],
        "status_changed": False,
        "tracked_diff_changed": False,
        "note": manifest["stability"]["note"],
    }

    sums = (out_dir / f"{manifest['snapshot_id']}.sha256sums.txt").read_text("utf-8").splitlines()
    assert len(sums) == len(packed)
    for line in sums:
        digest, path = line.split("  ", 1)
        assert digest == packed[path]["sha256"]

    fresh = tmp_path / "extracted"
    fresh.mkdir()
    _extract(package_path, fresh)
    assert (
        packer.main(
            ["verify", "--manifest", str(_load_manifest_path(out_dir)), "--tree", str(fresh)]
        )
        == 0
    )


def _load_manifest_path(out_dir: Path) -> Path:
    return sorted(out_dir.glob("*.manifest.json"))[0]


def test_clean_tree_is_refused_by_default(packer, tmp_path: Path) -> None:
    repo = _new_repo(tmp_path)
    out_dir = tmp_path / "packages"
    assert _build(packer, repo, out_dir) == 2
    assert not list(out_dir.glob("*.tar.gz"))
    assert _build(packer, repo, out_dir, "--no-require-uncommitted") == 0
    manifest = _load_manifest(out_dir)
    assert manifest["uncommitted_evidence"]["verified"] is False


def test_build_is_byte_deterministic_for_a_fixed_timestamp(packer, tmp_path: Path) -> None:
    repo = _new_repo(tmp_path)
    (repo / "src" / "owac" / "new.py").write_text("NEW = 1\n", encoding="utf-8")
    first, second = tmp_path / "a", tmp_path / "b"
    assert _build(packer, repo, first) == 0
    assert _build(packer, repo, second) == 0
    manifest_a, manifest_b = _load_manifest(first), _load_manifest(second)
    assert manifest_a["package"]["sha256"] == manifest_b["package"]["sha256"]
    assert manifest_a["files"] == manifest_b["files"]
    assert _sha256(first / f"{manifest_a['snapshot_id']}.tar.gz") == _sha256(
        second / f"{manifest_b['snapshot_id']}.tar.gz"
    )


def test_verify_detects_mismatch_missing_and_extras(packer, tmp_path: Path) -> None:
    repo = _new_repo(tmp_path)
    (repo / "src" / "owac" / "new.py").write_text("NEW = 1\n", encoding="utf-8")
    out_dir = tmp_path / "packages"
    assert _build(packer, repo, out_dir) == 0
    manifest = _load_manifest(out_dir)
    package_path = out_dir / f"{manifest['snapshot_id']}.tar.gz"

    tree = tmp_path / "tree"
    tree.mkdir()
    _extract(package_path, tree)
    manifest_path = str(_load_manifest_path(out_dir))

    assert packer.main(["verify", "--manifest", manifest_path, "--tree", str(tree)]) == 0

    (tree / "src" / "owac" / "mod.py").write_text("tampered\n", encoding="utf-8")
    (tree / "tests" / "test_mod.py").unlink()
    assert (
        packer.main(
            [
                "verify",
                "--manifest",
                manifest_path,
                "--tree",
                str(tree),
                "--package",
                str(package_path),
            ]
        )
        == 1
    )

    clean = tmp_path / "clean"
    clean.mkdir()
    _extract(package_path, clean)
    (clean / "stray.txt").write_text("extra\n", encoding="utf-8")
    assert packer.main(["verify", "--manifest", manifest_path, "--tree", str(clean)]) == 0
    assert (
        packer.main(
            ["verify", "--manifest", manifest_path, "--tree", str(clean), "--strict-extras"]
        )
        == 1
    )


def test_oversize_candidate_is_refused_not_dropped(packer, tmp_path: Path) -> None:
    repo = _new_repo(tmp_path)
    (repo / "src" / "owac" / "big.py").write_text("# pad\n" + "x = 1\n" * 400, encoding="utf-8")
    out_dir = tmp_path / "packages"
    assert _build(packer, repo, out_dir, "--max-file-bytes", "256") == 2
    assert (
        _build(
            packer,
            repo,
            out_dir,
            "--max-file-bytes",
            "256",
            "--allow-large-file",
            "src/owac/big.py",
        )
        == 0
    )
    manifest = _load_manifest(out_dir)
    record = {item["path"]: item for item in manifest["files"]}["src/owac/big.py"]
    assert record["large_file_override"] is True


def test_untracked_secret_like_paths_are_excluded(packer, tmp_path: Path) -> None:
    repo = _new_repo(tmp_path)
    (repo / "src" / "owac" / "credentials.json").write_text("{}\n", encoding="utf-8")
    (repo / "src" / "owac" / "server.pem").write_text("key\n", encoding="utf-8")
    (repo / "src" / "owac" / "regular_new.py").write_text("OK = 1\n", encoding="utf-8")
    out_dir = tmp_path / "packages"
    assert _build(packer, repo, out_dir) == 0
    manifest = _load_manifest(out_dir)
    packed = {item["path"] for item in manifest["files"]}
    assert "src/owac/regular_new.py" in packed
    assert "src/owac/credentials.json" not in packed
    assert "src/owac/server.pem" not in packed
    reasons = {item["path"]: item["reason"] for item in manifest["excluded"]}
    assert reasons["src/owac/credentials.json"] == "deny-name:credentials.json"
    assert reasons["src/owac/server.pem"] == "deny-secret-suffix:.pem"


def test_supervisor_template_renders_scoped_n04_program() -> None:
    rendered = TEMPLATE_PATH.read_text("utf-8")
    for token, value in PLACEHOLDERS.items():
        assert token in rendered, token
        rendered = rendered.replace(token, value)

    config = configparser.RawConfigParser()
    config.read_string(rendered)
    assert config.sections() == ["program:owac-n04-s1-demo-1"]
    program = config["program:owac-n04-s1-demo-1"]

    # nothing starts or restarts by itself: resume is an explicit decision
    assert program["autostart"] == "false"
    assert program["autorestart"] == "false"
    assert program["startretries"] == "0"

    # immutable snapshot as the code root, absolute interpreter, one CUDA device
    assert program["directory"] == PLACEHOLDERS["__SOURCE_DIR__"]
    assert program["command"].startswith(
        f"{PLACEHOLDERS['__PYTHON__']} -u -m owac.training.n04_full"
    )
    assert PLACEHOLDERS["__CONFIG__"] in program["command"]
    environment = program["environment"]
    assert f'CUDA_VISIBLE_DEVICES="{PLACEHOLDERS["__CUDA_DEVICE__"]}"' in environment
    assert f'PYTHONPATH="{PLACEHOLDERS["__SOURCE_DIR__"]}/src"' in environment
    assert f'N04_SNAPSHOT_ID="{PLACEHOLDERS["__SNAPSHOT_ID__"]}"' in environment

    # one run, foreground process, logs redirected into the run directory
    assert program["numprocs"] == "1"
    assert program["redirect_stderr"] == "true"
    assert program["stdout_logfile"].startswith(f"{PLACEHOLDERS['__RUN_DIR__']}/logs/")
    assert "__" not in program["command"]
    assert "__" not in program["stdout_logfile"]


def test_stability_flags_a_file_rewritten_during_packing(
    packer, tmp_path: Path, monkeypatch
) -> None:
    """Reproduces the live-checkout race: another writer replaces a packed file."""
    repo = _new_repo(tmp_path)
    target = repo / "src" / "owac" / "new.py"
    target.write_text("NEW = 1\n", encoding="utf-8")
    out_dir = tmp_path / "packages"
    original = packer.build_archive

    rewritten = "NEW = 2  # rewritten after packing\n"

    def build_then_rewrite(repo_path, staged, epoch, out_path):
        result = original(repo_path, staged, epoch, out_path)
        target.write_text(rewritten, encoding="utf-8")
        return result

    monkeypatch.setattr(packer, "build_archive", build_then_rewrite)
    assert _build(packer, repo, out_dir) == 0
    manifest = _load_manifest(out_dir)
    assert manifest["stability"]["stable"] is False
    assert manifest["stability"]["unstable_files"] == [
        {
            "path": "src/owac/new.py",
            "reason": "rewritten-after-packing",
            "bytes_now": len(rewritten.encode("utf-8")),
        }
    ]
    assert manifest["stability"]["tracked_diff_changed"] is False
    packed = {item["path"]: item for item in manifest["files"]}
    assert packed["src/owac/new.py"]["sha256"] == hashlib.sha256(b"NEW = 1\n").hexdigest()

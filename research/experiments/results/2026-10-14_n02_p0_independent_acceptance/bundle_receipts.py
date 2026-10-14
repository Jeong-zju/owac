"""Compose the current P0 evidence; never promote missing or partial checks."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

from owac.evaluation.n02_full import p0

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "outputs/n02-full/p0-independent-audit"
DEST = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    remote = read(RUN / "remote-summary.json")
    frozen = read(RUN / "frozen-data.json")
    for name, task_count in (("libero", 40), ("robotwin", 13)):
        row = remote["platforms"][name]
        assert row["task_count"] == task_count
        assert not row["verification"]["violations"]
        assert len(row["comparisons"]) == 5
        for sample in row["comparisons"]:
            assert sample["action_error"] == sample["state_error"] == 0
            assert max(sample["raw_errors"]) == 0
        assert frozen[name]["manifest_hash"] == row["manifest_hash"]
        for part in frozen[name]["partitions"].values():
            assert digest(ROOT / part["path"]) == part["gzip_sha256"]
    inventory = {
        "stage": "inventory",
        "status": "ok",
        "evidence_kind": "real_cache",
        "data_hash": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
        "platforms": remote["platforms"],
        "frozen_data": frozen,
        "scope": "all metadata and sizes; 5 real windows per platform; HF LFS hashes",
    }
    accepted = RUN / "accepted"
    for stage in ("inventory", "parity", "profile"):
        (accepted / stage).mkdir(parents=True, exist_ok=True)
    (accepted / "inventory/inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    shutil.copy2(RUN / "parity/parity.json", accepted / "parity/parity.json")
    shutil.copy2(RUN / "profile-accepted/profile.json", accepted / "profile/profile.json")
    code = p0.main(
        [
            "report",
            "--config",
            str(ROOT / "configs/encoders/n02_full/p0.json"),
            "--output",
            str(accepted / "report"),
        ]
    )
    assert code == 0
    for path in (accepted / "report").glob("*.json"):
        shutil.copy2(path, DEST / path.name)
    for name, path in {
        "inventory.json": accepted / "inventory/inventory.json",
        "parity.json": accepted / "parity/parity.json",
        "profile.json": accepted / "profile/profile.json",
        "profile-forward.json": RUN / "profile-forward-accepted/profile.json",
        "remote-summary.json": RUN / "remote-summary.json",
        "frozen-data.json": RUN / "frozen-data.json",
        "gen-upstream.json": RUN / "fixtures/gen_upstream.json",
        "sparse-upstream.json": RUN / "fixtures/fcmae_sparse_upstream.json",
        "theseus-upstream.json": RUN / "fixtures/theseus_upstream.json",
        "upstream-exports.json": RUN / "upstream-all.json",
        "inventory-libero.json": RUN / "inventory-libero.json",
        "inventory-robotwin.json": RUN / "inventory-robotwin.json",
    }.items():
        shutil.copy2(path, DEST / name)
    for name in ("libero", "robotwin"):
        raw = (RUN / f"manifest_{name}.json").read_bytes()
        (DEST / f"manifest-{name}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    source = {}
    for folder in (
        "src/owac/frameworks/n02_full",
        "src/owac/evaluation/n02_full",
        "tests/encoders/n02_full",
        "configs/encoders/n02_full",
        "configs/backends/n02-full-runtime",
    ):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                source[str(path.relative_to(ROOT))] = digest(path)
    for path in [
        ROOT / "src/owac/raw/rawvla_sequence.py",
        ROOT / "scripts/n02_p0_independent_cache_check.py",
    ]:
        source[str(path.relative_to(ROOT))] = digest(path)
    fixture_hashes = {p.name: digest(p) for p in (RUN / "fixtures").iterdir() if p.is_file()}
    (DEST / "source_manifest.json").write_text(
        json.dumps(
            {
                "git_head": p0.git_sha(),
                "working_tree_uncommitted": True,
                "files": source,
                "fixtures": fixture_hashes,
                "runtime_files": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in [
                        ROOT / "outputs/n02-full/runtime/julia-1.10.10.tar.gz",
                        next(
                            (
                                ROOT
                                / "third_party/pinned/shwoo93_MinkowskiEngine@bbc30ef581"
                                / "MinkowskiEngineBackend"
                            ).glob("*.so")
                        ),
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    for name in (
        "full-tests.log",
        "target-tests.log",
        "check.log",
        "sparse-validation.log",
        "gen-validation.log",
        "remote-recheck.log",
        "parity.log",
    ):
        shutil.copy2(RUN / name, DEST / name)
    assert read(DEST / "stage_decision.json")["implementation"] == "pass"
    print("Evidence bundle:", DEST)


if __name__ == "__main__":
    main()

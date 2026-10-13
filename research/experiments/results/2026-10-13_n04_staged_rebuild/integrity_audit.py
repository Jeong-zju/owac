"""Verify source, weight, paired data, identity, and artifact completeness."""

import hashlib
import json
import subprocess
from pathlib import Path

from run import ARCHIVE, digest, save


def main():
    protocols = {
        p.name.removesuffix("-protocol.json"): json.loads(p.read_text())
        for p in ARCHIVE.glob("s*-protocol.json")
    }
    assert len(protocols) == 6 and all(p["complete"] for p in protocols.values())
    for stage in ("s2-observation", "s2b-support", "s3a-imaging"):
        assert protocols[stage]["data_sha256"] == protocols["s1-data"]["data_sha256"]
    assert protocols["s3b-combined"]["data_sha256"] == protocols["s3c-combined-hint"]["data_sha256"]
    assert protocols["s3a-imaging"]["model_config"] == protocols["s3b-combined"]["model_config"]
    a, b = protocols["s3b-combined"]["model_config"], protocols["s3c-combined-hint"]["model_config"]
    assert {k for k in a if a[k] != b[k]} == {"illumination_explanation"}
    for left, right, field in (
        ("s1-data", "s2-observation", "innovation_reference"),
        ("s2-observation", "s2b-support", "match_observation_support"),
        ("s2b-support", "s3a-imaging", "illumination_explanation"),
    ):
        a, b = protocols[left]["model_config"], protocols[right]["model_config"]
        assert {
            k
            for k in set(a) | set(b)
            if a.get(k, False if k != "innovation_reference" else "task_state")
            != b.get(k, False if k != "innovation_reference" else "task_state")
        } == {field}
    weights = {}
    for stage, p in protocols.items():
        rows = json.loads((ARCHIVE / f"{stage}-matrix.json").read_text())
        assert len(rows) == p["suites"]
        for row in rows:
            cp = Path(row["checkpoint"])
            assert digest(cp) == row["checkpoint_sha256"]
            weights[str(cp)] = digest(cp)
    heldout = json.loads((ARCHIVE / "heldout-protocol.json").read_text())
    assert heldout["complete"]
    root = Path.cwd()
    for protocol in [*protocols.values(), heldout]:
        for source, sha in protocol["source_sha256"].items():
            path = Path(source)
            relative = str(path.relative_to(root)) if path.is_absolute() else source
            blob = subprocess.check_output(
                ["git", "show", f"{protocol['source_commit']}:{relative}"]
            )
            assert hashlib.sha256(blob).hexdigest() == sha
    for filename in (
        "combinations-provenance.json",
        "s3b-combined-oracles-motion-only-provenance.json",
        "s3b-combined-dynamics-development-provenance.json",
        "s3b-combined-dynamics-final-provenance.json",
    ):
        provenance = json.loads((ARCHIVE / filename).read_text())
        assert all(digest(Path(path)) == sha for path, sha in provenance["source_sha256"].items())
    rows = json.loads((ARCHIVE / "final-test-matrix.json").read_text())
    assert len(rows) == heldout["expected_suites"] == 54
    assert {(r["stage"], r["name"], r["run"], r["eval_seed"]) for r in rows} == {
        (stage, name, run, seed)
        for stage, name in heldout["targets"]
        for run in (0, 1, 2)
        for seed in heldout["seeds"]
    }
    for r in rows:
        assert digest(Path(r["checkpoint"])) == r["checkpoint_sha256"]
        assert len(r["report"]["conditions"]) == 13
        assert all(x["invariant_verified"] for x in r["report"]["counterfactuals"])
    for filename, count in [
        ("development-combinations.json", 216),
        ("s3b-combined-dynamics-development.json", 297),
        ("s3b-combined-dynamics-final.json", 297),
    ]:
        rows = json.loads((ARCHIVE / filename).read_text())
        assert len(rows) == count, (filename, len(rows))
        keys = (
            ("stage", "name", "run", "eval_seed", "condition")
            if filename.startswith("development")
            else ("name", "run", "seed", "kind", "mode")
        )
        assert len({tuple(r[k] for k in keys) for r in rows}) == count
    assert not subprocess.check_output(
        ["git", "ls-files", "-ci", "--exclude-standard"], text=True
    ).strip()
    hashes = {
        str(p): digest(p)
        for p in ARCHIVE.iterdir()
        if p.is_file() and p.name != "integrity-audit.json"
    }
    save(
        ARCHIVE / "integrity-audit.json",
        dict(
            complete=True,
            stage_count=len(protocols),
            checkpoints=len(weights),
            development_suites=sum(p["suites"] for p in protocols.values()),
            final_suites=54,
            verified_training_data_pairs=True,
            verified_single_factor_configs=True,
            verified_checkpoint_hashes=True,
            verified_complete_eval_identities=True,
            tracked_ignored_files=0,
            artifact_sha256=hashes,
        ),
    )
    print("audit passed", len(weights), "checkpoints")


if __name__ == "__main__":
    main()

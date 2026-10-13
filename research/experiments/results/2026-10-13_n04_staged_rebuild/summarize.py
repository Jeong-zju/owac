"""Aggregate paired independent runs and verify each frozen stage source/weight identity."""

import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ARCHIVE = Path("research/experiments/results/2026-10-13_n04_staged_rebuild")
KEYS = (
    "location_mean_px",
    "camera_vs_object_change",
    "illumination_response_ratio",
    "object_only/external_false_negative_rate",
    "both/external_false_negative_rate",
    "camera_only/external_false_positive_rate",
    "illumination/external_false_positive_rate",
    "camera_only/prediction_residual",
    "illumination/prediction_residual",
)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def metric(row, key):
    if key == "location_mean_px":
        return row[key]
    if "/" not in key:
        return row["report"]["contrast"][key]
    condition, field = key.split("/", 1)
    c = next(x for x in row["report"]["conditions"] if x["condition"] == condition)
    return c.get(field, c["extra"].get(field))


def aggregate(rows):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)
    out = {}
    for key in KEYS:
        values = {run: mean(metric(row, key) for row in group) for run, group in by_run.items()}
        present = [x for x in values.values() if x is not None]
        out[key] = dict(
            mean=mean(present),
            std_across_data_and_initialization=statistics.stdev(present)
            if len(present) > 1
            else None,
            per_run=values,
        )
    out["state_bytes_batch2"] = rows[0]["report"]["conditions"][0]["state_bytes"]
    out["condition_p50_mean_ms"] = mean(
        c["latency_p50_ms"] for r in rows for c in r["report"]["conditions"]
    )
    out["condition_p95_mean_ms"] = mean(
        c["latency_p95_ms"] for r in rows for c in r["report"]["conditions"]
    )
    out["recovery_censored_counts"] = {
        name: sum(
            c["extra"]["recovery_censored"]
            for r in rows
            for c in r["report"]["conditions"]
            if c["condition"] == name
        )
        for name in ("object_only", "occlusion")
    }
    return out


def main():
    summaries = {}
    audits = {}
    for protocol_path in sorted(ARCHIVE.glob("*-protocol.json")):
        protocol = json.loads(protocol_path.read_text())
        if "stage" not in protocol or not protocol["complete"]:
            continue
        stage = protocol["stage"]
        rows = json.loads((ARCHIVE / f"{stage}-matrix.json").read_text())
        assert len(rows) == protocol["suites"]
        expected = {
            (name, run, seed)
            for name in protocol["models"]
            for run in (0, 1, 2)
            for seed in protocol["development_seeds"]
        }
        assert {(r["name"], r["run"], r["eval_seed"]) for r in rows} == expected
        for source, sha in protocol["source_sha256"].items():
            blob = subprocess.check_output(["git", "show", f"{protocol['source_commit']}:{source}"])
            assert hashlib.sha256(blob).hexdigest() == sha, (stage, source)
        grouped = defaultdict(list)
        for row in rows:
            assert (
                hashlib.sha256(Path(row["checkpoint"]).read_bytes()).hexdigest()
                == row["checkpoint_sha256"]
            )
            assert all(x["invariant_verified"] for x in row["report"]["counterfactuals"])
            grouped[row["name"]].append(row)
        summaries[stage] = {name: aggregate(group) for name, group in grouped.items()}
        audits[stage] = dict(
            source_commit=protocol["source_commit"],
            verified_source_hashes=True,
            verified_weight_hashes=True,
            verified_paired_input_invariants=True,
            complete_development_identities=True,
            checkpoints=protocol["checkpoints"],
            suites=len(rows),
        )
    (ARCHIVE / "development-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (ARCHIVE / "stage-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    for stage, models in summaries.items():
        print(stage)
        for name, s in models.items():
            print(name, {k: s[k]["mean"] for k in KEYS})


if __name__ == "__main__":
    main()

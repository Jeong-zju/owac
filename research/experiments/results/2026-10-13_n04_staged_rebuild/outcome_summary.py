"""Summaries average evaluation scenes within each independently trained run first."""

import json
import statistics
from collections import defaultdict

from run import ARCHIVE, save


def stats(pairs):
    runs = defaultdict(list)
    for run, value in pairs:
        if value is not None:
            runs[run].append(value)
    means = {run: statistics.fmean(values) for run, values in runs.items()}
    values = list(means.values())
    return dict(
        mean=statistics.fmean(values) if values else None,
        std=statistics.stdev(values) if len(values) > 1 else None,
        per_run=means,
    )


def main():
    out = {}
    path = ARCHIVE / "final-test-matrix.json"
    if path.exists():
        rows = json.loads(path.read_text())
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["stage"], row["name"])].append(row)
        final = {}
        for (stage, name), group in grouped.items():
            summary = {}
            conditions = {c["condition"] for r in group for c in r["report"]["conditions"]}
            for condition in sorted(conditions):
                selected = [
                    (r["run"], c)
                    for r in group
                    for c in r["report"]["conditions"]
                    if c["condition"] == condition
                ]
                fields = {
                    k: stats((run, c[k]) for run, c in selected)
                    for k in (
                        "location_error",
                        "prediction_residual",
                        "change_response",
                        "warp_error",
                    )
                }
                fields.update(
                    {
                        k: stats((run, c["extra"][k]) for run, c in selected)
                        for k in (
                            "external_false_negative_rate",
                            "external_false_positive_rate",
                            "target_iou",
                        )
                    }
                )
                fields["censored"] = sum(c["extra"]["recovery_censored"] for _, c in selected)
                summary[condition] = fields
            for domain in ("original9", "combined3", "all13"):

                def include(c, domain=domain):
                    combo = c["condition"].endswith("_illumination")
                    return domain == "all13" or (
                        combo if domain == "combined3" else not combo and c["condition"] != "static"
                    )

                summary[domain] = stats(
                    (
                        r["run"],
                        statistics.fmean(
                            c["location_error"] for c in r["report"]["conditions"] if include(c)
                        ),
                    )
                    for r in group
                )
            for metric in ("camera_vs_object_change", "illumination_response_ratio"):
                summary[metric] = stats((r["run"], r["report"]["contrast"][metric]) for r in group)
            summary["state_bytes_batch2"] = group[0]["report"]["conditions"][0]["state_bytes"]
            for metric in ("latency_p50_ms", "latency_p95_ms"):
                summary[metric] = stats(
                    (r["run"], statistics.fmean(c[metric] for c in r["report"]["conditions"]))
                    for r in group
                )
            final[stage + "/" + name] = summary
        out["heldout"] = final
        # Paired contrasts retain each training run rather than treating scenes as repetitions.
        pairs = [
            ("s3b-combined/candidate", b)
            for b in (
                "s1-data/candidate",
                "s3b-combined/predict_update_filter",
                "s3b-combined/n04_no_change_branch",
                "s3c-combined-hint/candidate",
            )
        ]
        out["paired_location_delta"] = {
            a + " minus " + b: {
                domain: stats(
                    (run, value - final[b][domain]["per_run"][run])
                    for run, value in final[a][domain]["per_run"].items()
                )
                for domain in ("original9", "combined3", "all13")
            }
            for a, b in pairs
        }
    for split in ("development", "final"):
        path = ARCHIVE / f"s3b-combined-dynamics-{split}.json"
        if not path.exists():
            continue
        grouped = defaultdict(list)
        for row in json.loads(path.read_text()):
            grouped[(row["name"], row["mode"], row["kind"])].append(row)
        summaries = {}
        for (name, mode, kind), group in grouped.items():
            s = {
                k: stats((r["run"], r[k]) for r in group)
                for k in ("location_error", "change_response")
            }
            s.update(
                {
                    k: stats((r["run"], r["extra"][k]) for r in group)
                    for k in (
                        "hidden_error_px",
                        "after_event_error_px",
                        "external_false_positive_rate",
                        "external_false_negative_rate",
                    )
                }
            )
            s["recoveries"] = [r["recovery_frames"] for r in group]
            s["censored"] = sum(r["extra"]["recovery_censored"] for r in group)
            summaries[name + "/" + mode + "/" + kind] = s
        out["dynamics_" + split] = summaries
    path = ARCHIVE / "development-combinations.json"
    if path.exists():
        grouped = defaultdict(list)
        for r in json.loads(path.read_text()):
            grouped[(r["stage"], r["name"], r["condition"])].append(r)
        out["development_combinations"] = {
            "/".join(key): {
                "location": stats((r["run"], r["location_error"]) for r in group),
                "residual": stats((r["run"], r["prediction_residual"]) for r in group),
                "false_negative": stats(
                    (r["run"], r["extra"]["external_false_negative_rate"]) for r in group
                ),
                "false_positive": stats(
                    (r["run"], r["extra"]["external_false_positive_rate"]) for r in group
                ),
            }
            for key, group in grouped.items()
        }
    save(ARCHIVE / "outcome-summary.json", out)
    if "heldout" in out:
        for model, s in out["heldout"].items():
            print(model, {k: s[k]["mean"] for k in ("original9", "combined3", "all13")})
    if "dynamics_development" in out:
        for model, s in out["dynamics_development"].items():
            print(model, s["location_error"]["mean"], s["hidden_error_px"]["mean"])


if __name__ == "__main__":
    main()

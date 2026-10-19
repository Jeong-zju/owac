"""Recompute full r5 log statistics and reduce the read-only gradient receipts."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def correlation(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    dot = sum((v - mx) * (w - my) for v, w in zip(x, y, strict=True))
    norm = sum((v - mx) ** 2 for v in x) * sum((w - my) ** 2 for w in y)
    return dot / math.sqrt(norm) if norm else None


def main():
    parent = Path("research/experiments/n02_v1_2")
    destination = parent / "aux_audit_20261016"
    report, arms = {}, {}
    for arm in ("raw", "rgb"):
        source = parent / "training_completion_20261016" / arm
        opt = rows(source / "optimization.jsonl")
        hist = rows(source / "history.jsonl")
        arms[arm] = opt
        report[arm] = {
            "rows": len(opt),
            "contiguous": [r["step"] for r in opt] == list(range(1, 1001)),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (source / "optimization.jsonl", source / "history.jsonl")
            },
            "gradient_counts": {
                str(level): sum(r["grad_norm_before_clip"] > level for r in opt)
                for level in (1, 100, 1000)
            },
            "negative_causal_count": sum(r["terms"]["causal_raw"] < 0 for r in opt),
            "top_gradient_rows": sorted(
                opt, key=lambda r: r["grad_norm_before_clip"], reverse=True
            )[:10],
            "loss_stats": {
                key: {
                    "median": statistics.median(r["terms"][key] for r in opt),
                    "min": min(r["terms"][key] for r in opt),
                    "max": max(r["terms"][key] for r in opt),
                }
                for key in ("action", "masked_raw", "causal_raw")
            },
            "correlation_log_grad_with_log_abs_loss": {
                key: correlation(
                    [math.log(r["grad_norm_before_clip"]) for r in opt],
                    [math.log(abs(r["terms"][key]) + 1e-9) for r in opt],
                )
                for key in ("action", "masked_raw", "causal_raw")
            },
            "dev": {
                r["step"]: r["dev_action_error"]
                for r in hist
                if r.get("dev_action_error") is not None
            },
            "blocks": [
                {
                    "steps": [n, n + 99],
                    "median_grad": statistics.median(
                        r["grad_norm_before_clip"] for r in opt[n - 1 : n + 99]
                    ),
                    "median_terms": {
                        key: statistics.median(r["terms"][key] for r in opt[n - 1 : n + 99])
                        for key in ("action", "masked_raw", "causal_raw")
                    },
                }
                for n in range(1, 1001, 100)
            ],
        }
    report["paired_window_identity_all_steps"] = all(
        a["window"] == b["window"] for a, b in zip(arms["raw"], arms["rgb"], strict=True)
    )
    (destination / "log_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    summaries = []
    for path in sorted((destination / "receipts").glob("*/probe.json")):
        probe = json.loads(path.read_text())
        if "windows" not in probe:
            continue
        assert probe["weights_unchanged"] and probe["optimizer_updates"] == 0
        for row in probe["windows"]:
            full, encoder = row["roles"]["all"], row["roles"]["encoder"]
            virtual = row["virtual_adam"]
            item = {
                "model": path.parent.name,
                "window": row["label"],
                "action_l2": full["action_l2"],
                "causal_l2": full["causal_l2"],
                "causal_encoder_l2": encoder["causal_l2"],
                "action_encoder_l2": encoder["action_l2"],
                "masked_encoder_l2": encoder["masked_l2"],
                "cosine_action_masked_encoder": encoder["cosine_action_masked"],
                "delta_action_only": virtual["action_only"]["first_order_action_loss_change"],
                "delta_full": virtual["full"]["first_order_action_loss_change"],
                "clip_action_only": virtual["action_only"]["clip_coefficient"],
                "clip_full": virtual["full"]["clip_coefficient"],
                "action_q_path_fraction": row["action_path_l2"]["via_current_q"]
                / row["action_path_l2"]["full"],
            }
            item["descent_ratio_full_over_action_only"] = (
                item["delta_full"] / item["delta_action_only"]
            )
            summaries.append(item)
    (destination / "gradient_summary.json").write_text(
        json.dumps(
            {"rows": summaries, "optimizer_updates": 0, "all_parameters_unchanged": True}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "log_steps": sum(len(arm) for arm in arms.values()),
                "gradient_windows": len(summaries),
                "loss_ascent_first_order_windows": sum(r["delta_full"] > 0 for r in summaries),
            }
        )
    )


if __name__ == "__main__":
    main()

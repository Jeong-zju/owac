"""Completed bounded-run metrics and a trajectory-aware validation figure."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
summary = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
files = []
for arm, color in [("task", "#1b8064"), ("control", "#4268bc")]:
    folder = root / "receipts" / arm
    baseline = json.loads((folder / "dev_baseline.json").read_text())
    metrics = [json.loads(p.read_text()) for p in sorted(folder.glob("checkpoint_*.metrics.json"))]
    assert [m["step"] for m in metrics] == [16, 32, 48, 64]
    manifests = {m["task_dev_manifest_sha256"] for m in metrics}
    assert manifests == {baseline["task_dev_manifest_sha256"]}
    rows = [json.loads(s) for s in (folder / "optimization.jsonl").read_text().splitlines() if s]
    assert [r["step"] for r in rows] == list(range(1, 65))
    assert all(math.isfinite(r["terms"]["action"]) for r in rows)
    report = json.loads((folder / "report.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    initial = baseline["task_control"]["strata"]["Normal/opening"]["mse"]
    points = [
        dict(
            step=0,
            score=baseline["sampled_control_mse"],
            opening=initial,
            first10=baseline["task_control"]["strata"]["Normal/opening"]["first10_mse"],
        )
    ]
    points.extend(
        dict(
            step=m["step"],
            score=m["sampled_control_mse"],
            opening=m["task_control"]["strata"]["Normal/opening"]["mse"],
            first10=m["task_control"]["strata"]["Normal/opening"]["first10_mse"],
        )
        for m in metrics
    )
    for ax, key, denominator in [
        (axes[0], "score", baseline["sampled_control_mse"]),
        (axes[1], "opening", initial),
    ]:
        ax.plot(
            [p["step"] for p in points],
            [100 * (p[key] / denominator - 1) for p in points],
            marker="o",
            label=arm,
            color=color,
        )
        ax.grid(alpha=0.2)
        ax.set_xlabel("Optimizer updates")
        ax.spines[["top", "right"]].set_visible(False)
    summary[arm] = dict(
        points=points,
        selection=selection,
        overview=report["overview"],
        optimizer_updates=64,
        finite_action_loss=True,
        loss_min=min(r["terms"]["action"] for r in rows),
        loss_max=max(r["terms"]["action"] for r in rows),
    )
    files.extend(
        [
            folder / "dev_baseline.json",
            folder / "report.json",
            folder / "selection.json",
            folder / "optimization.jsonl",
        ]
    )
    files.extend(sorted(folder.glob("checkpoint_*.metrics.json")))
axes[0].set_title("Balanced sampled action error")
axes[1].set_title("Normal-light opening error")
axes[1].axhline(5, color="#b54132", ls="--", label="promotion limit")
for ax in axes:
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Change from own initialization (%)")
    ax.legend()
fig.savefig(root / "adaptation_results.png", dpi=180)
fig.savefig(root / "adaptation_results.svg")
(root / "results_summary.json").write_text(
    json.dumps(
        dict(
            arms=summary,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(summary, indent=2))

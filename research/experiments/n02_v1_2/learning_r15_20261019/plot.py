"""Scientific plots of audited fixed-budget action diagnostics."""

import json
import sys
from pathlib import Path

sys.path.append("/workspace/owac/rawvla/envs/robotwin/lib/python3.10/site-packages")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
report = json.loads((EVIDENCE / "summary.json").read_text())
rows = report["rows"]
names = {
    "state": "State only",
    "raw": "RAW + state",
    "native": "Native vision + state",
    "pi": "RAW adapter + pi",
}
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
x = np.arange(4)
for axis, split, title in (
    (axes[0], "train", "Training fit"),
    (axes[1], "normal_test", "Normal-light held-out"),
):
    for demos, offset, color in ((3, -0.18, "#73a598"), (5, 0.18, "#4777ab")):
        values = [
            next(
                r
                for r in rows
                if r["modality"] == m and r["demos"] == demos and r["budget"] == 1024
            )["selected"][split]["joint_rms_deg"]
            for m in names
        ]
        axis.bar(x + offset, values, 0.34, color=color, label=f"{demos} demonstration episodes")
        for xx, v in zip(x + offset, values, strict=True):
            axis.text(xx, v + 0.25, f"{v:.2f}", ha="center", fontsize=9)
    axis.set_xticks(x, list(names.values()), rotation=14)
    axis.set_title(title)
    axis.set_ylabel("Full 50-step joint RMS (degrees; lower is better)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.15)
axes[0].legend(fontsize=9)
fig.suptitle(
    "Matched small heads; separate pi adapter | 1024-update budget\n"
    "Validation-selected weights; frozen encoders; native RGB is original-light replay",
    fontsize=12,
)
fig.savefig(EVIDENCE / "modality_data_comparison.png", dpi=170)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), layout="constrained")
for axis, mode in zip(axes.flat, names, strict=True):
    for demos, color in ((3, "#2d8a73"), (5, "#5078b2")):
        chosen = sorted(
            [r for r in rows if r["modality"] == mode and r["demos"] == demos],
            key=lambda r: r["budget"],
        )
        xx = [r["budget"] for r in chosen]
        axis.plot(
            xx,
            [r["last"]["train"]["joint_rms_deg"] for r in chosen],
            ":",
            color=color,
            marker=".",
            label=f"{demos} demos / last training fit",
        )
        axis.plot(
            xx,
            [r["selected"]["val"]["joint_rms_deg"] for r in chosen],
            "--",
            color=color,
            label=f"{demos} demos / selected validation",
        )
        axis.plot(
            xx,
            [r["selected"]["normal_test"]["joint_rms_deg"] for r in chosen],
            "-o",
            color=color,
            label=f"{demos} demos / selected Normal held-out",
        )
    axis.set_title(names[mode])
    axis.set_xticks([256, 512, 1024])
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Joint RMS (degrees)")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.15)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8)
fig.suptitle(
    "Training progress and generalization are distinct\n"
    "Readouts only; no new closed-loop evaluations",
    fontsize=13,
)
fig.savefig(EVIDENCE / "learning_curves.png", dpi=170)

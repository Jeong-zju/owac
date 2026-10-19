"""Plot audited common action metrics, not incomparable training objectives."""

import json
import sys
from pathlib import Path

# Keep model runtime NumPy first; only append the existing plotting dependency.
sys.path.append("/workspace/owac/rawvla/envs/robotwin/lib/python3.10/site-packages")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EVIDENCE = Path(__file__).resolve().parent
report = json.loads((EVIDENCE / "summary.json").read_text())
rows = report["rows"]
labels = [r["branch"].upper() + " / " + r["phase"] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
x = np.arange(len(rows))
for axis, metric, title in (
    (axes[0], "normalized_control_mse", "Full 50-step action error"),
    (axes[1], "joint_rms_deg", "Joint error (degrees)"),
):
    for offset, split, label, color in (
        (-0.24, "selected_train", "Train (3 episodes)", "#337d71"),
        (0, "selected_val", "Validation (1 episode)", "#b89a3c"),
        (0.24, "selected_test", "Held-out dev (5 episodes)", "#657cb4"),
    ):
        axis.bar(x + offset, [r[split][metric] for r in rows], width=0.23, color=color, label=label)
    baseline = report["state_only_baseline"]["metrics"]["test"][metric]
    axis.axhline(baseline, color="#a95749", linestyle="--", label="State-only ridge / held-out")
    axis.set_xticks(x, labels, rotation=15)
    axis.set_title(title)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.15)
    axis.set_axisbelow(True)
axes[0].set_ylabel("Normalized control MSE (lower is better)")
axes[1].set_ylabel("Joint RMS in degrees (lower is better)")
axes[0].legend(fontsize=8)
fig.suptitle(
    "Same frozen RAW spatial tokens, then separate joint adaptation\n"
    "256 updates per branch per phase; no new closed-loop evaluation",
    fontsize=12,
)
fig.savefig(EVIDENCE / "action_comparison.png", dpi=170)

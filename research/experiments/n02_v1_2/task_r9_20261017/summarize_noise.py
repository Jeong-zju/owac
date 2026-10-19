"""Recompute ranges from measured noise probes and make an exportable scientific figure."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
sources = list(sorted((root / "receipts/noise").glob("d1-*/noise_probe.json")))
summary = {}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
for column, domain in enumerate(("Normal", "Low", "ExtremeLow")):
    ax = axes[column]
    for j, p in enumerate(sources):
        d = json.loads(p.read_text())
        rr = [r for r in d["rows"] if r["domain"] == domain]
        sensor = [r for r in rr if r["variant"] in ("raw", "redraw0", "redraw1", "redraw2")]
        flow = [r for r in rr if r["variant"] == "clean" and r["flow_seed"] != 188390828]
        for offset, rows, color in ((0, sensor, "#1b8064"), (1, flow, "#4268bc")):
            values = [r["joint_action_rms_difference_rad"] for r in rows]
            x = j * 3 + offset
            ax.scatter(np.full(len(values), x), values, color=color, s=32, alpha=0.8, zorder=3)
            ax.plot([x - 0.22, x + 0.22], [np.mean(values)] * 2, color=color, lw=2)
        summary[f"{p.parent.name}/{domain}"] = {
            "sensor_action_rms_rad_range": [
                min(r["joint_action_rms_difference_rad"] for r in sensor),
                max(r["joint_action_rms_difference_rad"] for r in sensor),
            ],
            "flow_action_rms_rad_range": [
                min(r["joint_action_rms_difference_rad"] for r in flow),
                max(r["joint_action_rms_difference_rad"] for r in flow),
            ],
            "sensor_token_relative_l2_range": [
                min(r["token_relative_l2"] for r in sensor),
                max(r["token_relative_l2"] for r in sensor),
            ],
        }
    ax.set_title(domain)
    ax.set_xticks([0, 1, 3, 4], ["Sensor\nnarrow", "Flow\nnarrow", "Sensor\nwide", "Flow\nwide"])
    ax.set_ylim(0, 0.35)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Joint action RMS change vs fixed reference (rad)")
fig.suptitle("Fixed initial observations: sensor noise vs action sampling", fontsize=13)
fig.savefig(root / "noise_sensitivity.png", dpi=180)
fig.savefig(root / "noise_sensitivity.svg")
(root / "noise_summary.json").write_text(
    json.dumps(
        dict(
            rows=summary,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            scope="one initial state per domain; action sensitivity, no success-rate claim",
        ),
        indent=2,
    )
    + "\n"
)

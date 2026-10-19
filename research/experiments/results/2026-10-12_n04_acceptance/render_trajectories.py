"""Run via uv run --no-project --with matplotlib python <this file>."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

root = Path(__file__).parent
payload = json.loads((root / "candidate-trajectory-seed10000.json").read_text())
conditions = ["camera_only", "object_only", "illumination", "calibration_error"]
figure, axes = plt.subplots(3, 4, figsize=(15, 7.5))
for col, name in enumerate(conditions):
    condition = next(c for c in payload["report"]["conditions"] if c["condition"] == name)
    rows = condition["trajectory"]
    times = [r["frame"] for r in rows]
    axes[0, col].set_title(name)
    for sample, color in enumerate(["#155e75", "#9f1239"]):
        for key, style, label in [("target_xy", "-", "truth"), ("predicted_xy", "--", "probe")]:
            axes[0, col].plot(
                [r[key][sample][0] for r in rows],
                [r[key][sample][1] for r in rows],
                style,
                color=color,
                marker="." if key == "target_xy" else None,
                markersize=3,
                label=f"{label} {sample}",
            )
    axes[0, col].set_aspect("equal", adjustable="datalim")
    axes[0, col].set_xlabel("x / px")
    axes[0, col].set_ylabel("y / px")
    axes[1, col].plot(times, [r["location_error_px"] for r in rows], color="#0f766e")
    axes[1, col].set_ylabel("Location error / px")
    axes[2, col].plot(
        times, [r["change_response"] for r in rows], label="change amplitude", color="#c2410c"
    )
    axes[2, col].plot(
        times, [r["unknown"] for r in rows], label="unknown (uncalibrated)", color="#7c3aed"
    )
    axes[2, col].set_ylabel("Amplitude / gate")
    for row in (1, 2):
        axes[row, col].set_xlabel("Frame")
    for row in range(3):
        axes[row, col].grid(alpha=0.2)
axes[0, 0].legend(fontsize=7)
axes[2, 0].legend(fontsize=7)
figure.suptitle(
    "N04 repaired tiny, train seed 0 / independent eval seed 10000\n"
    "Four trajectory panels; localization does not establish change separation",
    fontsize=12,
)
figure.tight_layout(rect=(0, 0, 1, 0.93))
figure.savefig(root / "four-trajectories.png", dpi=140)
figure.savefig(root / "four-trajectories.svg")
svg = root / "four-trajectories.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")

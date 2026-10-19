"""Plot actual target jumps and the independent roller-height replay receipts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
for name, path, color in [
    (
        "RAW100",
        Path("outputs/n02-v12/repair-r7-20261016/warm-step0100-attempt2/actions.jsonl"),
        "#ba4b30",
    ),
    (
        "Original pi0.5",
        Path("outputs/n02-v12/physics-audit-20261016/reference/actions.jsonl"),
        "#126c50",
    ),
]:
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    a = np.array([r["action"] for r in rows])[:, [*range(6), *range(7, 13)]]
    d = np.max(np.abs(np.diff(a, axis=0)), axis=1)
    ax[0].plot(np.arange(1, len(a)), d, label=name, color=color, alpha=0.8)
for x in (50, 100, 150, 200):
    ax[0].axvline(x, color="#777", alpha=0.2)
ax[0].set(
    xlabel="Control command index",
    ylabel="Largest joint target jump (rad)",
    title="Discontinuities at chunk boundaries",
)
ax[0].legend()
for mode, label, color in [
    ("raw", "RAW100 recorded predictions", "#ba4b30"),
    ("native", "Original pi0.5 recorded predictions", "#126c50"),
]:
    p = Path(f"outputs/n02-v12/physics-audit-20261016/{mode}-continued")
    rows = [json.loads(s) for s in (p / "physics.jsonl").read_text().splitlines()]
    summary = json.loads((p / "summary.json").read_text())
    trigger = summary["first_official_trigger"]
    t0 = trigger["physics_tick"] * trigger["dt"]
    t = np.array([r["physics_tick"] * r["dt"] - t0 for r in rows])
    ax[1].plot(t, [r["roller_p"][2] for r in rows], label=label, color=color)
ax[1].set(
    xlim=(-2, 18),
    ylim=(0, 0.95),
    xlabel="Simulation seconds relative to success flag",
    ylabel="Roller centre height (m)",
    title="A threshold crossing is not a completed grasp",
)
ax[1].axvline(0, color="#777", ls=":")
ax[1].axhline(0.8, color="#777", ls="--", lw=1)
ax[1].legend(fontsize=8)
fig.savefig("research/experiments/n02_v1_2/physics_audit_20261016/verification.png", dpi=150)
